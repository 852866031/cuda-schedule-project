#!/usr/bin/env python3
"""The colocated complementary workload: a small LoRA fine-tune on the decode GPU.

Trains LoRA adapters (r=16 on q,k,v,o) on a <1B causal LM with deterministic
synthetic batches, and logs one CSV row per optimizer step (wall-clock timestamp,
step time, tok/s, loss) so the trace can be aligned offline against the decode
run's window and the per-second GPU telemetry.

Run with .venv, pinned to the decode GPU:

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/inf_ft_coloc/ft_train.py \
        --out output/inf_ft_coloc/ft_solo.csv --duration 120

Safety: --mem-cap-gib (default 5.5) caps the torch caching allocator, so a
misconfigured trainer OOMs itself rather than starving a vLLM engine that has
not yet allocated its budget. Stops cleanly on SIGTERM/SIGINT (flushes the CSV).
"""

import argparse
import csv
import os
import signal
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # cached models only, never the network

import torch  # noqa: E402

TOTAL_GIB = 31.3536  # same constant as the sweep drivers: torch's total, not nvidia-smi's


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=0, help="0 = run until --duration or signal")
    ap.add_argument("--duration", type=float, default=0.0, help="seconds; 0 = unbounded")
    ap.add_argument("--mem-cap-gib", type=float, default=4.5)
    ap.add_argument("--duty", type=float, default=1.0,
                    help="fraction of wall-clock the trainer may occupy: after each "
                         "step of measured length dt it sleeps dt*(1-duty)/duty. "
                         "1.0 = train flat out (the plain-colocation arm)")
    ap.add_argument("--gate", action="store_true",
                    help="Orion-lite: before issuing each transformer block's kernels "
                         "(fwd + bwd) wait until the decode engine's busy page "
                         "(/dev/shm/coloc_hp_busy, written by orion_gate/hp_patch) "
                         "says the GPU is idle. Opens if the heartbeat goes stale.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(args.mem_cap_gib / TOTAL_GIB, dev)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    # GPT-2 fuses qkv into Conv1D "c_attn"; llama-family models use split projections.
    targets = (["c_attn", "c_proj"] if "gpt2" in args.model.lower()
               else ["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r,
        target_modules=targets, task_type="CAUSAL_LM"))
    model.to(dev).train()
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)

    vocab = model.config.vocab_size
    gen = torch.Generator().manual_seed(args.seed)

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))

    gate_stats = {"ns": 0}
    if args.gate:
        import mmap
        import struct
        fd = os.open("/dev/shm/coloc_hp_busy", os.O_CREAT | os.O_RDWR, 0o666)
        os.ftruncate(fd, 7 * 8)
        page = mmap.mmap(fd, 7 * 8)
        os.close(fd)
        MAGIC, STALE_NS = 0x434F4C4F43, 1_500_000_000

        # Bounded lookahead: CUDA issue is async, so an unsynchronized gate lets a
        # whole queue of blocks issued during one idle gap drain DURING the next
        # decode step. Waiting on the previous gated block's event first caps the
        # in-flight backlog at ~one block.
        fence = {"ev": None}

        def gate(*_):
            if struct.unpack_from("<q", page, 0)[0] != MAGIC:
                return
            t0 = time.monotonic_ns()
            if fence["ev"] is not None:
                fence["ev"].synchronize()
            while (struct.unpack_from("<q", page, 2 * 8)[0]
                   and time.monotonic_ns()
                   - struct.unpack_from("<q", page, 8)[0] < STALE_NS):
                time.sleep(0.0002)
            ev = torch.cuda.Event()
            ev.record()
            fence["ev"] = ev
            gate_stats["ns"] += time.monotonic_ns() - t0

        # Gate at transformer-block granularity, forward and backward: each block
        # is ~0.5-1 ms of kernels, the scale of the decode engine's idle windows.
        n_hooked = 0
        for m in model.modules():
            if type(m).__name__.endswith(("Block", "DecoderLayer")):
                m.register_forward_pre_hook(lambda *a: gate())
                m.register_full_backward_pre_hook(lambda *a: gate())
                n_hooked += 1
        print(f"ft_train: gating enabled on {n_hooked} blocks", flush=True)
    else:
        def gate(*_):
            pass

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    f = open(out, "w", newline="")
    w = csv.writer(f)
    w.writerow(["ts", "step", "dt_ms", "tok_per_s", "loss", "peak_gib", "gated_ms"])

    tokens = args.batch * args.seq
    t_start = time.time()
    step = 0
    print(f"ft_train: {args.model} lora_r={args.lora_r} batch={args.batch} seq={args.seq} "
          f"({tokens} tok/step), cap {args.mem_cap_gib} GiB", flush=True)
    while not stop["flag"]:
        if args.steps and step >= args.steps:
            break
        if args.duration and time.time() - t_start > args.duration:
            break
        ids = torch.randint(0, vocab, (args.batch, args.seq), generator=gen).to(dev)
        gate()
        gated0 = gate_stats["ns"]
        t0 = time.time()
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        gate()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = time.time() - t0
        step += 1
        peak = torch.cuda.max_memory_allocated(dev) / 1024**3
        w.writerow([f"{time.time():.3f}", step, f"{dt * 1e3:.1f}",
                    f"{tokens / dt:.0f}", f"{loss.item():.4f}", f"{peak:.2f}",
                    f"{(gate_stats['ns'] - gated0) / 1e6:.1f}"])
        if step % 20 == 0:
            f.flush()
            print(f"  step {step}: {dt * 1e3:.0f} ms, {tokens / dt:.0f} tok/s, "
                  f"loss {loss.item():.3f}, peak {peak:.2f} GiB", flush=True)
        if args.duty < 1.0:
            time.sleep(dt * (1.0 - args.duty) / args.duty)
    f.flush()
    f.close()
    print(f"ft_train: {step} steps in {time.time() - t_start:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
