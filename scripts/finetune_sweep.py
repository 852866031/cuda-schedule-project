#!/usr/bin/env python3
"""LoRA finetuning throughput as base-model layers are offloaded to DRAM.

Fixes LoRA rank 16 on attention (q,k,v,o) and the batch size, then sweeps how many of the 32
transformer layers keep their weights in VRAM. Offloaded layers are streamed to the GPU by
accelerate's hooks on every forward, and again on every backward.

Two entry points:

    python scripts/finetune_sweep.py --find-batch      # phase 0: largest power-of-2 batch that fits
    python scripts/finetune_sweep.py --batch 4         # the sweep, 0..32 layers offloaded by 4

Why a layer count rather than a memory fraction: layers are what accelerate actually places,
each is ~0.467 GiB, and the knob is exactly monotonic. The implied VRAM budget is not assumed
-- every point reports torch.cuda.max_memory_allocated().
"""

import argparse
import gc
import json
import subprocess
import time
from pathlib import Path

import torch

from layer_offload import OffloadManager

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "output"

MODEL = "NousResearch/Meta-Llama-3-8B-Instruct"
N_LAYERS = 32
SEQ_LEN = 2048
LORA_RANK = 16
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]  # attention only
PARAMS = 8.03e9
GIB = 1 << 30


def build_device_map(n_offload):
    """Put the last `n_offload` transformer layers on CPU, everything else on GPU 0.

    The tail is chosen rather than the head so the offloaded run still starts computing
    immediately -- the first layers are resident, which is also what a prefetching
    implementation would want.
    """
    dm = {
        "model.embed_tokens": 0,
        "model.norm": 0,
        "model.rotary_emb": 0,
        "lm_head": 0,
    }
    for i in range(N_LAYERS):
        dm[f"model.layers.{i}"] = "cpu" if i >= N_LAYERS - n_offload else 0
    return dm


def load(n_offload, batch, prefetch=False, depth=1, pattern="tail"):
    """Load fully onto the GPU, then hand the last `n_offload` layers to the streamer.

    Not device_map: accelerate's CPU offload leaves params on the meta device, which the
    backward pass rejects. Loading resident and then streaming out is equivalent in steady
    state and works through backward.
    """
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=LORA_RANK, lora_alpha=32, lora_dropout=0.0, bias="none",
        target_modules=LORA_TARGETS, task_type="CAUSAL_LM",
    ))
    # train() is load-bearing, not cosmetic: HF applies checkpointing only when
    # `self.gradient_checkpointing and self.training`, and from_pretrained returns a model in
    # eval mode. Without it the flag reads True, checkpointing silently does nothing, and
    # activations are ~14 GiB at 2048 tokens instead of ~1 GiB.
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()  # inputs to checkpointed blocks must be non-leaf

    mgr = None
    if n_offload:
        mgr = OffloadManager(model.base_model.model.model.layers, n_offload,
                             prefetch=prefetch, depth=depth, pattern=pattern)
        gc.collect()
        torch.cuda.empty_cache()
    return model, mgr


def synthetic_batch(batch, device):
    """Random token ids. This measures throughput, not learning -- the data is irrelevant,
    and random ids avoid any tokenizer/dataset variability between configs."""
    ids = torch.randint(128, 127_000, (batch, SEQ_LEN), device=device)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids}


def pcie_counters():
    """Cumulative PCIe throughput counters, KB. Differencing these gives bytes moved."""
    try:
        q = subprocess.run(
            ["nvidia-smi", "-i", "0", "--query-gpu=pcie.link.gen.current",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        return {"gen": q.stdout.strip()}
    except Exception:
        return {}


def run_point(n_offload, batch, steps, warmup, lr=1e-4, prefetch=False, depth=1,
              pattern="tail"):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model, mgr = load(n_offload, batch, prefetch, depth, pattern)
    load_s = time.time() - t0

    # Reset AFTER load: the model is loaded fully resident and then streamed out, so the
    # load-time peak (14.96 GiB) would otherwise mask the training peak at high offload and
    # make the VRAM column read a flat 15.02 GiB regardless of how much was offloaded.
    torch.cuda.reset_peak_memory_stats()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    data = synthetic_batch(batch, "cuda")

    step_times, losses = [], []
    for i in range(warmup + steps):
        torch.cuda.synchronize()
        s = time.time()
        if mgr is not None:
            mgr.pre_forward()
        out = model(**data)
        if mgr is not None:
            mgr.backward(out.loss)
        else:
            out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = time.time() - s
        if i >= warmup:
            step_times.append(dt)
            losses.append(float(out.loss))
        print(f"    step {i:2d}{' (warmup)' if i < warmup else ''}: {dt:6.2f}s", flush=True)

    step_times.sort()
    median = step_times[len(step_times) // 2]
    tokens = batch * SEQ_LEN
    peak_gib = torch.cuda.max_memory_allocated() / GIB
    # LoRA freezes the base weights, so the backward computes input gradients but not weight
    # gradients: 4*N*T FLOPs, not the usual 6*N*T. Denominator is the RTX 5090's bf16 dense
    # peak (~209 TFLOPS), so this is a true model-FLOPs-utilisation figure.
    mfu = 4 * PARAMS * tokens / median / 209e12

    rec = {
        "n_offload": n_offload,
        "prefetch": prefetch,
        "prefetch_depth": depth if prefetch else 0,
        "pattern": pattern,
        "n_resident": N_LAYERS - n_offload,
        "batch": batch, "seq_len": SEQ_LEN, "tokens_per_step": tokens,
        "lora_rank": LORA_RANK, "trainable_params": trainable,
        "load_s": round(load_s, 1),
        "step_s_median": round(median, 3),
        "step_s_min": round(step_times[0], 3),
        "step_s_max": round(step_times[-1], 3),
        "tokens_per_s": round(tokens / median, 1),
        "mfu": round(mfu, 4),
        "peak_vram_gib": round(peak_gib, 2),          # training only, load excluded
        "resident_weight_gib": round((PARAMS * 2 / GIB) * (N_LAYERS - n_offload) / N_LAYERS, 2),
        "offloaded_weight_gib": round(mgr.offloaded_bytes / GIB, 2) if mgr else 0.0,
        "layer_fetches": mgr.fetches if mgr else 0,
        "loss_mean": round(sum(losses) / len(losses), 4),
        "steps": steps,
    }
    if mgr is not None:
        mgr.remove()
    del model, opt, data, mgr
    gc.collect()
    torch.cuda.empty_cache()
    return rec


def find_batch(candidates, steps=2):
    """Largest power-of-2 batch that trains with all layers resident, with headroom."""
    print("PHASE 0: finding the largest batch that fits with 0 layers offloaded\n")
    results = []
    for b in candidates:
        print(f"  trying batch {b} ({b * SEQ_LEN:,} tokens/step)...", flush=True)
        try:
            rec = run_point(0, b, steps=steps, warmup=1)
            headroom = 31.35 - rec["peak_vram_gib"]
            print(f"    FITS: peak {rec['peak_vram_gib']} GiB, headroom {headroom:.2f} GiB, "
                  f"{rec['tokens_per_s']} tok/s\n", flush=True)
            results.append(rec)
            return b, results          # descending order, so the first that fits is the answer
        except torch.cuda.OutOfMemoryError:
            print("    OOM\n", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}\n", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    return None, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--find-batch", action="store_true")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--offload-step", type=int, default=4)
    ap.add_argument("--max-offload", type=int, default=N_LAYERS,
                    help="highest number of layers to offload (default: all 32)")
    ap.add_argument("--pattern", choices=["tail", "interleave"], default="tail",
                    help="which layers to offload: the last N, or every stride-th")
    ap.add_argument("--prefetch-depth", type=int, default=1,
                    help="how many layers ahead to stage (1 = next layer only)")
    ap.add_argument("--prefetch", action="store_true",
                    help="overlap the next layer's copy with the current layer's compute")
    ap.add_argument("--tag", default="ft")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")

    if args.find_batch or args.batch is None:
        batch, probes = find_batch([16, 8, 4, 2, 1])
        (OUT / f"{args.tag}_batch_probe.json").write_text(json.dumps(probes, indent=2))
        if batch is None:
            print("no batch size fits -- investigate")
            return
        print(f"=> chosen batch: {batch} ({batch * SEQ_LEN:,} tokens/step)")
        if args.find_batch:
            return
    else:
        batch = args.batch

    rows = []
    for n in range(0, args.max_offload + 1, args.offload_step):
        print(f"\n=== {n} layers offloaded ({N_LAYERS - n} resident) ===", flush=True)
        try:
            rec = run_point(n, batch, args.steps, args.warmup, prefetch=args.prefetch,
                            depth=args.prefetch_depth, pattern=args.pattern)
            print(f"  -> {rec['step_s_median']}s/step, {rec['tokens_per_s']} tok/s, "
                  f"MFU {rec['mfu']:.1%}, peak {rec['peak_vram_gib']} GiB", flush=True)
        except Exception as e:
            rec = {"n_offload": n, "batch": batch, "error": f"{type(e).__name__}: {e}"}
            print(f"  FAILED: {rec['error']}", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
        rows.append(rec)
        (OUT / f"{args.tag}_sweep.json").write_text(json.dumps(rows, indent=2))

    print(f"\nwrote output/{args.tag}_sweep.json")


if __name__ == "__main__":
    main()
