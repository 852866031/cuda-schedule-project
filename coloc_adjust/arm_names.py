"""Plain-language names for the evaluation arms (shared by the figure scripts)."""

NAMES = {
    "decode_alone":    "Inference alone, fixed 13 GiB KV",
    "adjust_notenant": "Inference alone, KV follows load (baseline)",
    "static_split":    "Static split: KV 6.7 GiB, FT 6 GiB",
    "adjust":          "Adaptive KV + FT waits for idle GPU (gate)",
    "adjust_nogate":   "Adaptive KV + FT shares GPU via MPS",
    "adjust_mps20":    "Adaptive KV + FT capped at 20 % of SMs (MPS)",
    "green32":         "Adaptive KV + FT on a 32-SM green context",
    "green_dyn":       "Adaptive KV + FT SM count from load (dynamic green context)",
    "engine_green":    "Adaptive KV + engine on 128 SMs, FT dynamic green context",
    "green_opt":       "Adaptive KV + smoothed shrink, lower FT SM table",
    "green_split":     "Adaptive KV + disjoint SMs: engine 128, FT 42",
}

# tenant configuration index -> short label (ft_worker.py TIERS)
CFG_LABELS = ["full batch", "k=2", "k=4", "k=8", "+act offload", "+12 layers", "+20 layers", "paused"]


def name(arm):
    return NAMES.get(arm, arm)
