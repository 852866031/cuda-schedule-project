#!/usr/bin/env python3
import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output"
PLOT_DIR = ROOT / "output" / "plots"
OVERHEAD_PLAIN_JSON = ROOT / "output" / "llm_metrics.json"
OVERHEAD_TRACE_JSON = ROOT / "output" / "llm_trace_metrics.json"
GLOBAL_CSV = OUT_DIR / "kernel_hotspots_global.csv"
RECENT_CSV = OUT_DIR / "kernel_hotspots_recent_5s.csv"

TOP_K = 10


def load_rows(csv_path: Path):
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "kernel_name": row["kernel_name"],
                    "launch_count": int(row["launch_count"]),
                    "total_duration_ms": float(row["total_duration_ms"]),
                    "avg_duration_us": float(row["avg_duration_us"]),
                }
            )
    return rows

def load_overhead_rows(csv_path: Path):
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "mode": row["mode"],
                    "elapsed_sec": float(row["elapsed_sec"]),
                }
            )
    return rows

def humanize_kernel_name(name: str) -> str:
    s = name.strip()

    patterns = [
        ("fmha_cutlass", "Flash/MemEff Attention"),
        ("cutlass::Kernel2", "CUTLASS GEMM"),
        ("gemm", "GEMM"),
        ("reduce_kernel", "Reduction"),
        ("CatArrayBatchedCopy_vectorized", "Tensor Concat Copy (Vec)"),
        ("CatArrayBatchedCopy", "Tensor Concat Copy"),
        ("direct_copy_kernel", "Direct Copy"),
        ("vectorized_elementwise_kernel", "Vectorized Elementwise"),
        ("unrolled_elementwise_kernel", "Unrolled Elementwise"),
        ("elementwise_kernel", "Elementwise"),
        ("layer_norm", "LayerNorm"),
        ("softmax", "Softmax"),
        ("embedding", "Embedding"),
    ]

    for pat, label in patterns:
        if pat in s:
            return label

    s = s.replace("void ", "")
    s = s.replace("at::native::(anonymous namespace)::", "")
    s = s.replace("at::native::", "")
    s = s.replace("cutlass::", "")
    if len(s) > 50:
        s = s[:47] + "..."
    return s


def shorten_raw_name(name: str, max_len: int = 120) -> str:
    return name if len(name) <= max_len else name[: max_len - 3] + "..."


def make_display_names_and_notes(rows):
    display_names = []
    footnotes = []
    label_counts = {}

    for i, r in enumerate(rows, start=1):
        base = humanize_kernel_name(r["kernel_name"])
        label_counts[base] = label_counts.get(base, 0) + 1

        if label_counts[base] == 1:
            disp = f"{base} [{i}]"
        else:
            disp = f"{base} #{label_counts[base]} [{i}]"

        display_names.append(disp)
        footnotes.append(f"[{i}] {shorten_raw_name(r['kernel_name'])}")

    return display_names, footnotes


def plot_one(csv_path: Path, out_path: Path, title_prefix: str):
    rows = load_rows(csv_path)
    rows = sorted(rows, key=lambda x: x["total_duration_ms"], reverse=True)[:TOP_K]
    rows.reverse()

    display_names, footnotes = make_display_names_and_notes(rows)
    duration_vals = [r["total_duration_ms"] for r in rows]
    count_vals = [r["launch_count"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    ax0, ax1 = axes

    ax0.barh(display_names, duration_vals)
    ax0.set_title(f"{title_prefix}: Total Duration")
    ax0.set_xlabel("Total duration (ms)")
    ax0.set_ylabel("Kernel")

    ax1.barh(display_names, count_vals)
    ax1.set_title(f"{title_prefix}: Launch Count")
    ax1.set_xlabel("Launch count")
    ax1.set_ylabel("Kernel")

    footnote_text = "\n".join(footnotes)

    fig.subplots_adjust(bottom=0.34, wspace=0.35)
    fig.text(
        0.01,
        0.02,
        footnote_text,
        ha="left",
        va="bottom",
        fontsize=8,
        family="monospace",
    )

    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved plot to {out_path}")

def load_mode_value_csv(csv_path: Path, key_name: str):
    rows = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["mode"]] = float(row[key_name])
    return rows

def plot_overhead(plain_json_path: Path, trace_json_path: Path, out_path: Path):
    import json

    with plain_json_path.open("r", encoding="utf-8") as f:
        plain = json.load(f)
    with trace_json_path.open("r", encoding="utf-8") as f:
        trace = json.load(f)

    plain_time = float(plain["total_time_sec"])
    trace_time = float(trace["total_time_sec"])

    plain_tps = float(plain["tokens_per_sec"])
    trace_tps = float(trace["tokens_per_sec"])

    time_overhead_sec = trace_time - plain_time
    time_overhead_pct = (time_overhead_sec / plain_time * 100.0) if plain_time > 0 else 0.0

    tps_delta = trace_tps - plain_tps
    tps_delta_pct = (tps_delta / plain_tps * 100.0) if plain_tps > 0 else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax0, ax1 = axes

    labels = ["llm", "llm_trace"]

    # Left: total time
    time_vals = [plain_time, trace_time]
    time_bars = ax0.bar(labels, time_vals)
    ax0.set_title("Total Execution Time")
    ax0.set_ylabel("Time (s)")

    for bar, val in zip(time_bars, time_vals):
        ax0.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.3f}s",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax0.set_ylim(0, max(time_vals) * 1.25 if max(time_vals) > 0 else 1.0)
    ax0.annotate(
        f"Overhead = {time_overhead_sec:.3f}s ({time_overhead_pct:.1f}%)",
        xy=(1, trace_time),
        xytext=(0.5, max(time_vals) * 1.12),
        textcoords="data",
        ha="center",
        fontsize=10,
    )

    # Right: tokens per second
    tps_vals = [plain_tps, trace_tps]
    tps_bars = ax1.bar(labels, tps_vals)
    ax1.set_title("Tokens per Second")
    ax1.set_ylabel("Tokens/s")

    for bar, val in zip(tps_bars, tps_vals):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax1.set_ylim(0, max(tps_vals) * 1.25 if max(tps_vals) > 0 else 1.0)
    ax1.annotate(
        f"Δ tokens/s = {tps_delta:.2f} ({tps_delta_pct:.1f}%)",
        xy=(1, trace_tps),
        xytext=(0.5, max(tps_vals) * 1.12),
        textcoords="data",
        ha="center",
        fontsize=10,
    )

    fig.suptitle("LLM Runtime With vs Without Tracer", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    print(f"Saved plot to {out_path}")
    
def main():
    plot_one(
        GLOBAL_CSV,
        PLOT_DIR / "kernel_hotspots_global.png",
        "Global Hotspots",
    )
    plot_one(
        RECENT_CSV,
        PLOT_DIR / "kernel_hotspots_recent_5s.png",
        "Recent 5s Hotspots",
    )
    plot_overhead(
        OVERHEAD_PLAIN_JSON,
        OVERHEAD_TRACE_JSON,
        ROOT / "output" / "plots" / "overhead_compare.png",
    )


if __name__ == "__main__":
    main()