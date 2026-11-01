#!/usr/bin/env python3
"""E3(b): can a decode step stream its KV from host memory instead of holding it?

One Llama-3-8B-shaped decode step (32 layers; per layer RMSNorm-free QKV / O / gate-up /
down GEMMs at the real shapes, GQA attention of 1 query token over a 6400-token context)
for batch B. Arms:

  resident   K/V for all layers live in HBM (what vLLM does)
  stream d   each layer's K/V is copied from pinned host memory during the step, on a
             second stream, up to d layers ahead of the compute (a ring of d+1 HBM
             buffers). d = 0: copy layer l, then compute layer l (no overlap).
  copy       the copies alone (B x 0.78 GiB per step): the PCIe floor

The host side reuses ONE layer-sized pinned buffer as the source for every layer: the
bytes moved over PCIe are the same as with 32 distinct buffers, and pinned memory stays
at B x 25 MiB (400 MiB at B = 16) -- no pinned overcommit risk. Weights are random bf16
(32 distinct layers, 14 GiB). Step time = median of N CUDA-event-timed steps.

    .venv/bin/python scripts/kv_vram/bench_stream.py --gpu 1   # -> output/kv_vram/bench_stream.json
"""
import argparse
import json
import os
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=1)
ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 12, 16])
ap.add_argument("--depths", type=int, nargs="+", default=[0, 1, 2, 4, 8])
ap.add_argument("--ctx", type=int, default=6400)
ap.add_argument("--iters", type=int, default=10)
ap.add_argument("--out", default="output/kv_vram/bench_stream.json")
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import torch                                    # noqa: E402
import torch.nn.functional as F                 # noqa: E402

L, H, NH, KVH, D, FF = 32, 4096, 32, 8, 128, 14336
dt = torch.bfloat16
dev = torch.device("cuda")
torch.manual_seed(0)


def w(*shape):
    return (torch.randn(*shape, device=dev, dtype=dt) * 0.02)


W = [dict(qkv=w(H, (NH + 2 * KVH) * D), o=w(NH * D, H), gu=w(H, 2 * FF), dn=w(FF, H))
     for _ in range(L)]


def layer(x, k, v, p):
    """x: (B, H). k, v: (B, KVH, ctx, D)."""
    B = x.shape[0]
    qkv = x @ p["qkv"]
    q = qkv[:, :NH * D].view(B, NH, 1, D)
    a = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
    x = x + a.reshape(B, NH * D) @ p["o"]
    g, u = (x @ p["gu"]).chunk(2, dim=-1)
    return x + (F.silu(g) * u) @ p["dn"]


def timed(fn, iters):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return round(ts[len(ts) // 2], 3)


def run_batch(B):
    x0 = torch.randn(B, H, device=dev, dtype=dt) * 0.1
    kv_shape = (B, KVH, args.ctx, D)
    layer_bytes = 2 * B * KVH * args.ctx * D * 2
    row = {"batch": B, "kv_gib_per_step": round(L * layer_bytes / 2**30, 3)}

    # resident: 32 layers of K/V in HBM
    try:
        Kr = [torch.randn(kv_shape, device=dev, dtype=dt) for _ in range(L)]
        Vr = [torch.randn(kv_shape, device=dev, dtype=dt) for _ in range(L)]

        def resident():
            x = x0
            for i in range(L):
                x = layer(x, Kr[i], Vr[i], W[i])
            return x
        row["resident_ms"] = timed(resident, args.iters)
        del Kr, Vr
    except torch.OutOfMemoryError:
        row["resident_ms"] = None
    torch.cuda.empty_cache()

    # host source: one pinned layer (K and V), reused for every layer
    hk = torch.randn(kv_shape, dtype=dt).pin_memory()
    hv = torch.randn(kv_shape, dtype=dt).pin_memory()
    cs = torch.cuda.Stream()

    def copy_only():
        bk = torch.empty(kv_shape, device=dev, dtype=dt)
        bv = torch.empty(kv_shape, device=dev, dtype=dt)
        for _ in range(L):
            bk.copy_(hk, non_blocking=True)
            bv.copy_(hv, non_blocking=True)
    row["copy_ms"] = timed(copy_only, args.iters)

    for d in args.depths:
        nbuf = max(1, d + 1) if d > 0 else 1
        nbuf = min(nbuf, L)
        bk = [torch.empty(kv_shape, device=dev, dtype=dt) for _ in range(nbuf)]
        bv = [torch.empty(kv_shape, device=dev, dtype=dt) for _ in range(nbuf)]

        if d == 0:
            def streamed():
                x = x0
                for i in range(L):
                    bk[0].copy_(hk, non_blocking=True)
                    bv[0].copy_(hv, non_blocking=True)
                    x = layer(x, bk[0], bv[0], W[i])
                return x
        else:
            def streamed():
                main = torch.cuda.current_stream()
                loaded = [torch.cuda.Event() for _ in range(L)]
                freed = [torch.cuda.Event() for _ in range(L)]

                def issue(j):
                    with torch.cuda.stream(cs):
                        if j >= nbuf:
                            cs.wait_event(freed[j - nbuf])   # slot reuse after compute
                        s = j % nbuf
                        bk[s].copy_(hk, non_blocking=True)
                        bv[s].copy_(hv, non_blocking=True)
                        loaded[j].record(cs)
                cs.wait_stream(main)
                for j in range(min(d, L)):
                    issue(j)
                x = x0
                for i in range(L):
                    if i + d < L:
                        issue(i + d)
                    main.wait_event(loaded[i])
                    s = i % nbuf
                    x = layer(x, bk[s], bv[s], W[i])
                    freed[i].record(main)
                main.wait_stream(cs)
                return x
        row[f"stream_d{d}_ms"] = timed(streamed, args.iters)
        del bk, bv
        torch.cuda.empty_cache()
    del hk, hv
    return row


rows = []
t0 = time.time()
for B in args.batches:
    r = run_batch(B)
    rows.append(r)
    print(json.dumps(r), flush=True)
out = {"gpu_index": args.gpu, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
       "ctx": args.ctx, "layers": L, "iters": args.iters, "rows": rows,
       "wall_s": round(time.time() - t0, 1)}
p = Path(__file__).resolve().parents[2] / args.out
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(out, indent=1))
print("->", p)
