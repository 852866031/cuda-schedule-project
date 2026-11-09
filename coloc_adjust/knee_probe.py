#!/usr/bin/env python
"""How many SMs does a decode step need? One synthetic decode step with Llama-3-8B shapes (32 layers:
merged qkv, o, merged gate_up, down; GQA attention over a per-request KV; lm_head on the last token),
random weights (timing does not depend on values), run on green contexts of different SM counts.
A decode step moves ~15 GB of weights plus 0.82 GB of KV per running request and only B tokens of
compute, so it should stop speeding up once the SMs can saturate GPU-memory bandwidth: that SM count
is the knee. A 2048-token prefill chunk (compute-bound) is the contrast.

    CUDA_VISIBLE_DEVICES=1 python knee_probe.py --out results/knee_probe.csv
"""
import argparse, csv, math, time
import torch, torch.nn.functional as F
from green_ctx import GreenCtx

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results/knee_probe.csv")
ap.add_argument("--sms", default="8,16,24,32,40,48,56,64,72,80,96,112,128,144,160,0", help="SM counts (the driver rounds to multiples of 8); 0 = full device (default stream)")
ap.add_argument("--batches", default="1,4,8,14", help="running requests in the decode step")
ap.add_argument("--ctx", type=int, default=6272, help="context tokens per request (0.82 GB of KV)")
ap.add_argument("--prefill-tokens", type=int, default=2048)
ap.add_argument("--reps", type=int, default=5)
args = ap.parse_args()

H, I, L, NKV, GRP, HD, V = 4096, 14336, 32, 8, 4, 128, 128256      # Llama-3-8B: 32 heads = 8 KV heads x 4
dev, dt = "cuda", torch.bfloat16
BS = [int(b) for b in args.batches.split(",")]; BMAX = max(BS)
def w(*shape): return torch.empty(*shape, device=dev, dtype=dt).normal_(0, 0.02)
t0 = time.time()
W = [dict(qkv=w(H + 2 * NKV * HD, H), o=w(H, H), gate_up=w(2 * I, H), down=w(H, I)) for _ in range(L)]
LM = w(V, H)
KT = [w(BMAX, NKV, HD, args.ctx) for _ in range(L)]        # keys stored transposed: scores = q @ KT, no hidden copy
VV = [w(BMAX, NKV, args.ctx, HD) for _ in range(L)]
W_GB = (sum(sum(t.numel() for t in d.values()) for d in W) + LM.numel()) * 2 / 1e9
KV_GB_PER_REQ = 2 * L * NKV * HD * args.ctx * 2 / 1e9
print(f"weights {W_GB:.2f} GB, KV {KV_GB_PER_REQ:.3f} GB per request, built in {time.time()-t0:.1f}s, "
      f"allocated {torch.cuda.memory_allocated()/2**30:.1f} GiB", flush=True)

@torch.inference_mode()
def step(x, B, n_new, ctx):
    """x: [B*n_new, H]. Every request attends over `ctx` cached tokens; returns the next token per request."""
    for l in range(L):
        d = W[l]
        q = F.linear(F.rms_norm(x, (H,)), d["qkv"])[:, :H]
        q = q.reshape(B, n_new, NKV, GRP, HD).permute(0, 2, 1, 3, 4).reshape(B, NKV, n_new * GRP, HD)
        kt, vv = (KT[l][:B], VV[l][:B]) if ctx == args.ctx else (KT[l][:B, :, :, :ctx].contiguous(), VV[l][:B, :, :ctx].contiguous())
        a = torch.softmax((q @ kt) * (HD ** -0.5), dim=-1) @ vv                      # [B, 8, n_new*4, 128]
        a = a.reshape(B, NKV, n_new, GRP, HD).permute(0, 2, 1, 3, 4).reshape(B * n_new, H)
        x = x + F.linear(a, d["o"])
        gu = F.linear(F.rms_norm(x, (H,)), d["gate_up"])
        x = x + F.linear(F.silu(gu[:, :I]) * gu[:, I:], d["down"])
    last = x.reshape(B, n_new, H)[:, -1]
    return F.linear(F.rms_norm(last, (H,)), LM).argmax(-1)

CONFIGS = [("decode", b, 1, args.ctx) for b in BS] + [("prefill", 1, args.prefill_tokens, args.prefill_tokens)]
rows = []
for n in [int(s) for s in args.sms.split(",")]:
    g = GreenCtx(n) if n else None
    stream = g.torch_stream() if g else torch.cuda.default_stream()
    sms = g.sm_count if g else torch.cuda.get_device_properties(0).multi_processor_count
    with torch.cuda.stream(stream):
        for kind, B, n_new, ctx in CONFIGS:
            x = w(B * n_new, H)
            t = time.perf_counter(); step(x, B, n_new, ctx); stream.synchronize(); step(x, B, n_new, ctx); stream.synchronize()
            t1 = (time.perf_counter() - t) / 2
            steps = max(1, min(20, int(0.5 / t1)))
            l1 = []                                   # CPU time to launch one step into an empty queue: must stay below step_ms
            for _ in range(3):
                stream.synchronize(); t = time.perf_counter(); step(x, B, n_new, ctx); l1.append(time.perf_counter() - t)
            launch1 = sorted(l1)[1] * 1e3; stream.synchronize()
            for rep in range(args.reps):
                t = time.perf_counter()
                for _ in range(steps): out = step(x, B, n_new, ctx)
                tl = time.perf_counter(); stream.synchronize(); torch.cuda.synchronize(); te = time.perf_counter()
                gb = W_GB + B * KV_GB_PER_REQ * ctx / args.ctx
                rows.append(dict(sms_req=n, sms=sms, kind=kind, B=B, tokens=B * n_new, ctx=ctx, rep=rep, steps=steps,
                                 step_ms=(te - t) / steps * 1e3, launch1_ms=launch1, gb_per_step=gb))
            med = sorted(r["step_ms"] for r in rows[-args.reps:])[args.reps // 2]
            print(f"SMs {sms:3d}  {kind:7s} B={B:2d} tokens={B*n_new:5d}  step {med:8.2f} ms  launch of one step {launch1:6.2f} ms  "
                  f"{rows[-1]['gb_per_step']/med*1e3:6.0f} GB/s", flush=True)
            del x, out
    if g: g.close()
    torch.cuda.empty_cache()
with open(args.out, "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
print("wrote", args.out)
