#!/usr/bin/env python3
"""Synthetic multi-session workload with an exactly-known KV working set.

Real datasets can't pin the KV working set to a target size, and this experiment needs it
pinned: the whole point is to shrink the VRAM budget *relative to a fixed working set*.

Construction:
  - `num_sessions` sessions, each a unique random prefix of `prefix_len` tokens.
  - every request = one session's prefix + a short unique suffix -> the prefix is reusable,
    the suffix never is, so each request has one cacheable prefix and one real prefill.
  - sessions are chosen by a Zipf or uniform draw, which sets how much of the working set
    is "hot".

Prompts are sent as raw token IDs, not text. Text would be re-tokenized by the server and
the prefix length would drift; token IDs give exact, block-aligned control.

KV geometry (Llama-3-8B fp16): 32 layers * 8 kv heads * 128 dim * 2 (K,V) * 2 bytes
                             = 131072 B/token = 0.125 MiB/token.

Default shape: 32 sessions * 6144 tokens = 196608 tokens = exactly 24.0 GiB of KV.
6144 + 128 suffix + 128 output = 6400 < 8192, so it fits Llama-3's context window.
"""

from dataclasses import dataclass, field

import numpy as np

KV_BYTES_PER_TOKEN = 32 * 8 * 128 * 2 * 2  # 131072
GIB = 1 << 30

# Llama-3 reserves 128000-128255 for special tokens; stay well clear of them and of id 0.
VOCAB_LO, VOCAB_HI = 128, 127_000

# vLLM V1 default block size. Prefixes are kept block-aligned so a session's KV maps onto
# whole cache blocks -- a partial tail block is never cached and would silently cost a
# re-prefill on every hit.
BLOCK_SIZE = 16


@dataclass
class Request:
    index: int
    session_id: int
    token_ids: list = field(repr=False)
    prompt_len: int


@dataclass
class Workload:
    sessions: list = field(repr=False)
    requests: list = field(repr=False)
    num_sessions: int
    prefix_len: int
    suffix_len: int
    skew: str
    zipf_a: float
    seed: int

    @property
    def working_set_tokens(self) -> int:
        return self.num_sessions * self.prefix_len

    @property
    def working_set_gib(self) -> float:
        return self.working_set_tokens * KV_BYTES_PER_TOKEN / GIB

    @property
    def prefix_gib(self) -> float:
        """KV of a single session prefix -- the unit that moves over PCIe on a DRAM hit."""
        return self.prefix_len * KV_BYTES_PER_TOKEN / GIB

    def summary(self) -> dict:
        counts = np.bincount([r.session_id for r in self.requests],
                             minlength=self.num_sessions)
        order = np.argsort(counts)[::-1]
        cum = np.cumsum(counts[order]) / max(1, counts.sum())
        # How many distinct sessions cover 90% of requests -> the "hot set" whose KV the
        # GPU tier would need to hold to avoid most DRAM traffic.
        hot_90 = int(np.searchsorted(cum, 0.90) + 1)
        return {
            "num_sessions": self.num_sessions,
            "prefix_len": self.prefix_len,
            "suffix_len": self.suffix_len,
            "num_requests": len(self.requests),
            "skew": self.skew,
            "zipf_a": self.zipf_a,
            "seed": self.seed,
            "working_set_tokens": self.working_set_tokens,
            "working_set_gib": round(self.working_set_gib, 3),
            "prefix_kv_gib": round(self.prefix_gib, 4),
            "hot_sessions_90pct": hot_90,
            "hot_set_90pct_gib": round(hot_90 * self.prefix_gib, 3),
            "session_request_counts": counts.tolist(),
        }


def build_workload(
    num_sessions: int = 32,
    prefix_len: int = 6144,
    suffix_len: int = 128,
    num_requests: int = 300,
    skew: str = "zipf",
    zipf_a: float = 1.1,
    seed: int = 0,
) -> Workload:
    if prefix_len % BLOCK_SIZE:
        raise ValueError(f"prefix_len {prefix_len} must be a multiple of {BLOCK_SIZE}")

    rng = np.random.default_rng(seed)
    sessions = [
        rng.integers(VOCAB_LO, VOCAB_HI, size=prefix_len, dtype=np.int64).tolist()
        for _ in range(num_sessions)
    ]

    if skew == "uniform":
        picks = rng.integers(0, num_sessions, size=num_requests)
    elif skew == "zipf":
        # Rank-ordered Zipf over exactly `num_sessions` sessions. np.random.zipf is
        # unbounded, so build the truncated pmf directly instead of rejection-sampling.
        ranks = np.arange(1, num_sessions + 1)
        pmf = ranks.astype(np.float64) ** (-zipf_a)
        pmf /= pmf.sum()
        picks = rng.choice(num_sessions, size=num_requests, p=pmf)
    else:
        raise ValueError(f"unknown skew {skew!r} (expected 'zipf' or 'uniform')")

    requests = []
    for i, sid in enumerate(picks):
        suffix = rng.integers(VOCAB_LO, VOCAB_HI, size=suffix_len, dtype=np.int64).tolist()
        token_ids = sessions[int(sid)] + suffix
        requests.append(Request(i, int(sid), token_ids, len(token_ids)))

    return Workload(sessions, requests, num_sessions, prefix_len, suffix_len,
                    skew, zipf_a, seed)


def warmup_requests(wl: Workload) -> list:
    """One request per session, so every session's KV exists in some tier before measuring.

    Without this the first touch of each session is a cold prefill and the early part of a
    run measures cache-filling rather than steady-state serving.
    """
    out = []
    rng = np.random.default_rng(wl.seed + 99991)
    for sid in range(wl.num_sessions):
        suffix = rng.integers(VOCAB_LO, VOCAB_HI, size=wl.suffix_len, dtype=np.int64).tolist()
        out.append(Request(-1 - sid, sid, wl.sessions[sid] + suffix,
                           wl.prefix_len + wl.suffix_len))
    return out


if __name__ == "__main__":
    import json
    for skew in ("zipf", "uniform"):
        wl = build_workload(skew=skew)
        s = wl.summary()
        s.pop("session_request_counts")
        print(json.dumps(s, indent=2))
