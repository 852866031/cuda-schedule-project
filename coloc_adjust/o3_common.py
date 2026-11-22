"""Shared by the overnight-3 drivers (corun2.py, tenant_bytes.py): tenant variants of C2 and the counter calibration."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
O3 = os.path.join(HERE, "results_o3")
# name -> ft_worker.py flags. "Same training": the result of a step is unchanged. "Less training": fewer layers are adapted.
VARIANTS = {
    "base": [],
    "eager": ["--attn", "eager"],
    "compile": ["--compile"],
    "losschunk": ["--loss-chunk", "16"],
    "compile_losschunk": ["--compile", "--loss-chunk", "16"],
    "ckpt": ["--force-ckpt"],
    "top8": ["--lora-top-layers", "8"],
    "top4": ["--lora-top-layers", "4"],
    "fwd": ["--forward-only"],
    # plan of Nov 21 (names as in results_prof/bytes_per_token_prof.json): the output layer over vocabulary blocks; the same with every block compiled and the block's operators fused
    "vocab4096": ["--head", "vocab", "--vocab-block", "4096", "--bpt-variant", "vocab4096"],
    "cb_vocab4096f": ["--compile-blocks", "--no-slow-shapes", "--head", "vocab", "--vocab-block", "4096", "--head-fused", "--bpt-variant", "cb_vocab4096f"],
}
SAME_TRAINING = ["compile", "losschunk", "compile_losschunk"]
LESS_TRAINING = ["top8", "top4"]
KIND = {"base": "usual", "eager": "same training (check)", "ckpt": "same training", "fwd": "reference, no training",
        **{v: "same training" for v in SAME_TRAINING}, **{v: "less training" for v in LESS_TRAINING}}
def gb_per_pct(default=17.79):
    """GB/s per 1 % of the device-wide bandwidth counter: tonight's calibration (corun2.py --plan r0), else Nov 9's"""
    try: return float(json.load(open(os.path.join(O3, "calib.json")))["gb_per_pct"])
    except Exception: return default
def chosen_variants():
    try: return json.load(open(os.path.join(O3, "r2_variants.json")))
    except Exception: return {"V1": None, "V2": "top4"}
