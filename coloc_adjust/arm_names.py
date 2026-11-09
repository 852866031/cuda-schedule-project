"""Plain-language names for the evaluation schemes (the --arm values) (shared by the figure scripts)."""

NAMES = {
    "decode_alone":    "Inference alone, fixed 13 GiB KV",
    "adjust_notenant": "Inference alone, KV follows load (baseline)",
    "static_split":    "Static split: KV 6.7 GiB, FT 6 GiB",
    "static_split_gate": "Static split with the idle gate (tenant starves)",
    "adjust":          "Adaptive KV + FT waits for idle GPU (gate)",
    "adjust_nogate":   "Adaptive KV + FT shares GPU via MPS",
    "adjust_mps20":    "Adaptive KV + FT capped at 20 % of SMs (MPS)",
    "green32":         "Adaptive KV + FT on a 32-SM green context",
    "green_dyn":       "Adaptive KV + FT SM count from load (dynamic green context)",
    "engine_green":    "Adaptive KV + engine on 128 SMs, FT dynamic green context",
    "green_opt":       "Adaptive KV + smoothed shrink, lower FT SM table",
    "green_split":     "Adaptive KV + disjoint SMs: engine 128, FT 42",
    "tpot_slo5":       "Adaptive KV + FT throttled to a 5 % decode-latency SLO",
    "tpot_slo15":      "Adaptive KV + FT throttled to a 15 % decode-latency SLO",
    "green_dyn2":      "Adaptive KV + FT SM count from load (prebuilt green contexts)",
    "pf_base":         "Inference alone, KV follows load, 25.5 GiB host tier (prefill-scheme baseline)",
    "pf_mps":          "Adaptive KV + 3B prefill tenant, shared SMs (MPS)",
    "pf_mps20":        "Adaptive KV + 3B prefill tenant capped at 20 % of SMs (MPS)",
    "pf_split":        "Adaptive KV + 3B prefill tenant on disjoint SMs: engine 128, tenant 42",
}

# tenant configuration index -> short label (ft_worker.py TIERS)
CFG_LABELS = ["full batch", "k=2", "k=4", "k=8", "+act offload", "+12 layers", "+20 layers", "paused"]


def name(arm):
    return NAMES.get(arm, arm)

# scheme id -> (legend label, color, marker). Order = legend order.
ARMS = [
    ("adjust_notenant", "Inference alone, KV follows load (baseline)", "#444444", "o"),
    ("static_split",    "Static split: KV 6.7 GiB, FT 6 GiB",          "#d62728", "s"),
    ("adjust",          "Adaptive KV + FT waits for idle GPU (gate)",   "#9467bd", "^"),
    ("adjust_nogate",   "Adaptive KV + FT shares GPU via MPS",          "#1f77b4", "v"),
    ("adjust_mps20",    "Adaptive KV + FT capped at 20 % of SMs (MPS)", "#2ca02c", "D"),
    ("green_split",     "Adaptive KV + disjoint SMs: engine 128, FT 42", "#ff7f0e", "P"),
    ("tpot_slo5",       "Adaptive KV + FT throttled to a 5 % latency SLO",  "#8c564b", "X"),
    ("tpot_slo15",      "Adaptive KV + FT throttled to a 15 % latency SLO", "#e377c2", "*"),
    ("green_dyn2",      "Adaptive KV + FT SM count from load (prebuilt green ctx)", "#17becf", "h"),
]
# the prefill-tenant schemes get their own figure (plot_arms.py --pf): different tenant, different throughput unit
PF_ARMS = [
    ("pf_base",         "Inference alone, KV follows load, 25.5 GiB host tier (baseline)", "#444444", "o"),
    ("pf_mps",          "3B prefill tenant, shared SMs (MPS)",           "#1f77b4", "v"),
    ("pf_mps20",        "3B prefill tenant capped at 20 % of SMs (MPS)", "#2ca02c", "D"),
    ("pf_split",        "3B prefill tenant on disjoint SMs (42 of 170)", "#ff7f0e", "P"),
]
BASE = "adjust_notenant"
PF_BASE = "pf_base"
