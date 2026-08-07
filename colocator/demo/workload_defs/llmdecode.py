"""llmdecode — real LLM decode steps: Llama-3-8B, 64-token KV cache, 3 passes.

The decode phase of LLM inference, isolated: a 63-token random prompt
(fixed seed) determines the KV cache, and token 64 is the pending input.
Each timed iteration is exactly one decode forward pass — one new token in,
attention over the (64, 65, 66)-entry cache, greedy argmax out — i.e. the
generation of tokens 65, 66 and 67. drain() restores the post-prefill cache
snapshot, so warmup and the timed loop replay the SAME three decode steps
instead of growing the cache run-long.

THE PREFILL RUNS AT MOST ONCE PER MACHINE, not once per run: the first run
computes it and persists the post-prefill state (KV tensors + pending
token, deterministic given seed+weights) to ~/.cache/colocator/; later runs
load that snapshot and execute PURE DECODE — no prefill kernels on the
timeline, no cuBLAS split-K/cutlass burst, nothing but the decode passes.
Delete the snapshot file (or bump SNAP_VERSION) to force a re-prefill.

Sampling stays on the GPU (argmax fed back as the next input, no .item()), so
a pass launches its kernels without any host sync and the depth-2 pipeline of
the base class works unchanged.

INTERCEPTION: the linear layers (q/k/v/o, MLP, lm_head — 85% of decode GPU
time) go through cuBLAS, whose internal kernel launches are invisible to
LD_PRELOAD (statically-linked runtime / driver API). The colocator therefore
intercepts the cuBLAS *API call* instead (cublasGemmEx / cublasSgemm_v2 —
Orion-style; see src/intercept), queues it like any op, and the scheduler
replays it with the stream substituted on the handle. That makes this
workload colocatable: every op — aten kernel, cuBLAS GEMV, memcpy — flows
through the same per-client queue in program order onto one colocator
stream. check() (greedy decode must be deterministic) doubles as the
colocated-correctness test.

Setup notes: after the weights land we cudaDeviceSynchronize before
prefilling — under the colocator, load copies and compute could otherwise
overlap across streams. setup() also stamps marks["load_done"], which the
visualizer uses to trim the ~1.3 s weight upload from solo timelines.
"""

import copy
import os
import time

import torch

from base import WorkloadBase

MODEL_ID = "meta-llama/Meta-Llama-3-8B"
CACHE_TOKENS = 64   # cache entries attended by the FIRST decode pass
DECODE_STEPS = 3    # tokens 65, 66, 67
SNAP_VERSION = 1    # bump to invalidate persisted prefill snapshots
SNAP_PATH = os.path.expanduser(
    f"~/.cache/colocator/llmdecode-{MODEL_ID.split('/')[-1]}"
    f"-{CACHE_TOKENS}tok-v{SNAP_VERSION}.pt")


class LlmDecodeClient(WorkloadBase):
    name = "llmdecode"

    def __init__(self, iters=None, device="cuda:0"):
        super().__init__(iters=iters, device=device)
        if self.iters is None:
            self.iters = DECODE_STEPS  # the workload IS 3 decode passes; no ~500ms calibration

    def setup(self):
        from transformers import AutoModelForCausalLM  # deferred: heavy import
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16).to(self.device).eval()
        # Device-wide sync: the load's copies must be complete before compute
        # (colocated, copies and kernels would otherwise cross streams), and
        # the mark must postdate them so timeline trimming is exact.
        torch.cuda.synchronize()
        self.marks["load_done"] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        # Snapshot: cache of tokens 1..63 + token 64 as the pending input. The
        # first decode pass appends token 64's KV, attends over 64 entries and
        # produces token 65 — matching "decode with a 64-token KV cache".
        # Loaded from disk when a previous run already prefilled (the state is
        # deterministic: fixed seed + fixed weights); recomputed and persisted
        # otherwise. weights_only=False: it's our own local pickle.
        try:
            snap = torch.load(SNAP_PATH, map_location=self.device,
                              weights_only=False)
            self._snap_cache, self._snap_tok = snap["cache"], snap["tok"]
            self.prefilled = "loaded snapshot"
        except Exception:  # missing file or stale pickle format → re-prefill
            cfg = self.model.config
            torch.manual_seed(0)
            ids = torch.randint(0, cfg.vocab_size, (1, CACHE_TOKENS),
                                device=self.device)
            with torch.no_grad():
                out = self.model(input_ids=ids[:, :CACHE_TOKENS - 1], use_cache=True)
            self._snap_cache = out.past_key_values
            self._snap_tok = ids[:, CACHE_TOKENS - 1:]
            torch.cuda.current_stream().synchronize()
            os.makedirs(os.path.dirname(SNAP_PATH), exist_ok=True)
            torch.save({"cache": self._snap_cache, "tok": self._snap_tok},
                       SNAP_PATH)
            self.prefilled = "prefilled + saved snapshot"
        self._reset()

    def _reset(self):
        # deepcopy clones the per-layer KV tensors — API-version-agnostic, and
        # small (~2·layers·kv_heads·head_dim·64 tokens·bf16 ≈ 8 MiB for 8B).
        self.cache = copy.deepcopy(self._snap_cache)
        self.next_tok = self._snap_tok.clone()

    def launch_iter(self):
        with torch.no_grad():
            out = self.model(input_ids=self.next_tok, past_key_values=self.cache,
                             use_cache=True)
            self._logits = out.logits[:, -1, :]
            self.next_tok = self._logits.argmax(-1, keepdim=True)  # greedy, stays on GPU
        return self.next_tok

    def drain(self):
        super().drain()
        if hasattr(self, "_snap_cache"):
            self._reset()

    def check(self):
        # Greedy decode from a restored snapshot must be deterministic, and
        # the logits finite. (.item() syncs — fine here, check isn't timed.)
        t1 = int(self.run_iter_sync().item())
        finite = bool(torch.isfinite(self._logits).all().item())
        t2 = int(self.run_iter_sync().item())
        ok = finite and t1 == t2 and 0 <= t1 < self.model.config.vocab_size
        if not ok:
            print(f"[{self.name}] check failed: tok1={t1} tok2={t2} finite={finite}")
        return ok

    def summary(self):
        return (f"Real <b>LLM decode</b> steps: {MODEL_ID} (bf16). The post-prefill state of "
                f"a {CACHE_TOKENS - 1}-token random prompt (computed once per machine, "
                f"persisted in ~/.cache/colocator) is loaded in setup; each timed iteration "
                f"is one decode forward pass — generating tokens {CACHE_TOKENS + 1}…"
                f"{CACHE_TOKENS + DECODE_STEPS} with a KV cache that grows "
                f"{CACHE_TOKENS}→{CACHE_TOKENS + DECODE_STEPS - 1} entries. The cache is "
                f"restored to the snapshot at every drain, so every loop replays the same "
                f"{DECODE_STEPS} steps.")

    def params_html(self):
        cfg = getattr(self, "model", None) and self.model.config
        if not cfg:
            return ""
        kv_dim = cfg.num_key_value_heads * (cfg.hidden_size // cfg.num_attention_heads)
        cache_mib = 2 * cfg.num_hidden_layers * CACHE_TOKENS * kv_dim * 2 / (1 << 20)
        return (f"<li>{cfg.num_hidden_layers} layers · hidden {cfg.hidden_size} · "
                f"{cfg.num_attention_heads} heads / {cfg.num_key_value_heads} KV heads (GQA) · "
                f"vocab {cfg.vocab_size}</li>"
                f"<li>per pass: 7 cuBLAS GEMV/GEMMs ×{cfg.num_hidden_layers} layers + lm_head, "
                f"SDPA attention over ≤{CACHE_TOKENS + DECODE_STEPS - 1} cached tokens, "
                f"RMSNorm/RoPE elementwise</li>"
                f"<li>KV cache at {CACHE_TOKENS} tokens ≈ {cache_mib:.0f} MiB · "
                f"weights ≈ {sum(p.numel() for p in self.model.parameters()) * 2 >> 30} GiB bf16</li>"
                f"<li>linear layers = cuBLAS GEMVs, intercepted at the <b>cublasGemmEx</b> "
                f"API call (kernel-level interception is impossible for cuBLAS) and "
                f"replayed by the scheduler with the stream substituted</li>"
                f"<li>prefill state: <b>{getattr(self, 'prefilled', '?')}</b> — with a "
                f"loaded snapshot the run executes decode passes only</li>")


WORKLOAD = LlmDecodeClient
