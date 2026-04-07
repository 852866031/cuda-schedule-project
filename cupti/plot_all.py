#!/usr/bin/env python3
import csv
import json
import glob as glob_mod
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output"
PLOT_DIR = ROOT / "output" / "plots"
OVERHEAD_PLAIN_JSON = OUT_DIR / "llm_metrics.json"
OVERHEAD_TRACE_JSON = OUT_DIR / "llm_trace_metrics.json"
OVERHEAD_PROFILE_JSON = OUT_DIR / "llm_profile_metrics.json"
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
    """Two-way overhead comparison: plain vs tracer."""
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


# =========================================================================
# Three-way overhead comparison: plain vs tracer vs profiler
# =========================================================================

def plot_overhead_three_way(
    plain_json_path: Path,
    trace_json_path: Path,
    profile_json_path: Path,
    out_path: Path,
):
    """Compare execution time and throughput across plain / tracer / profiler."""
    jsons = {}
    for label, path in [
        ("llm", plain_json_path),
        ("llm_trace", trace_json_path),
        ("llm_profile", profile_json_path),
    ]:
        if not path.exists():
            print(f"  Skipping three-way overhead plot: {path} not found")
            return
        with path.open("r", encoding="utf-8") as f:
            jsons[label] = json.load(f)

    labels = list(jsons.keys())
    colors = ["#4c72b0", "#55a868", "#c44e52"]

    times = [float(jsons[l]["total_time_sec"]) for l in labels]
    tps_vals = [float(jsons[l]["tokens_per_sec"]) for l in labels]
    base_time = times[0]
    base_tps = tps_vals[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax0, ax1 = axes

    # --- Left: total execution time ---
    bars0 = ax0.bar(labels, times, color=colors)
    ax0.set_title("Total Execution Time")
    ax0.set_ylabel("Time (s)")
    ax0.set_ylim(0, max(times) * 1.30 if max(times) > 0 else 1.0)

    for bar, val, label in zip(bars0, times, labels):
        overhead_s = val - base_time
        overhead_pct = (overhead_s / base_time * 100.0) if base_time > 0 else 0.0
        text = f"{val:.2f}s"
        if label != "llm":
            text += f"\n(+{overhead_pct:.1f}%)"
        ax0.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            text,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # --- Right: tokens per second ---
    bars1 = ax1.bar(labels, tps_vals, color=colors)
    ax1.set_title("Tokens per Second")
    ax1.set_ylabel("Tokens/s")
    ax1.set_ylim(0, max(tps_vals) * 1.30 if max(tps_vals) > 0 else 1.0)

    for bar, val, label in zip(bars1, tps_vals, labels):
        delta = val - base_tps
        delta_pct = (delta / base_tps * 100.0) if base_tps > 0 else 0.0
        text = f"{val:.1f}"
        if label != "llm":
            text += f"\n({delta_pct:+.1f}%)"
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            text,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.suptitle(
        "LLM Runtime: Plain vs Tracer vs Profiler", fontsize=13, fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


# =========================================================================
# Per-cycle profiling results plots
# =========================================================================

def _find_profile_cycle_jsons(out_dir: Path):
    """Return a sorted list of profile_cycle_*.json paths."""
    paths = sorted(out_dir.glob("profile_cycle_*.json"))
    return paths


def plot_profile_cycle(json_path: Path, out_path: Path):
    """
    Generate a two-panel figure for a single profiling cycle:
      Left:  horizontal bar chart of all collected metrics
      Right: kernel identity card (name, launches, avg duration, trace window)
    """
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    cycle = data.get("cycle", "?")
    kernel_name = data.get("target_kernel", "unknown")
    trace_s = data.get("trace_duration_s", "?")
    launches = data.get("total_launches", 0)
    total_ms = data.get("total_gpu_time_ms", 0.0)
    avg_us = data.get("avg_duration_us", 0.0)
    metrics = data.get("metrics", {})

    if not metrics:
        print(f"  Skipping {json_path.name}: no metrics collected")
        return

    metric_names = list(metrics.keys())
    metric_values = [metrics[k] for k in metric_names]

    # Shorten metric names for display
    short_names = []
    for m in metric_names:
        # e.g. "sm__cycles_active.avg" -> "sm__cycles_active .avg"
        short_names.append(m if len(m) <= 45 else m[:42] + "...")

    fig, axes = plt.subplots(
        1, 2, figsize=(16, max(4.5, 1.2 * len(metric_names))),
        gridspec_kw={"width_ratios": [3, 2]},
    )
    ax_bar, ax_info = axes

    # --- Left: metric values bar chart ---
    colors = []
    for m in metric_names:
        if "dram" in m:
            colors.append("#c44e52")
        elif "warp" in m or "occupancy" in m:
            colors.append("#dd8452")
        else:
            colors.append("#4c72b0")

    y_pos = range(len(short_names))
    ax_bar.barh(y_pos, metric_values, color=colors)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(short_names, fontsize=9, family="monospace")
    ax_bar.set_xlabel("Value")
    ax_bar.set_title(f"Profiling Metrics (cycle {cycle})", fontsize=12)
    ax_bar.invert_yaxis()

    # Add value labels on bars
    max_val = max(metric_values) if metric_values else 1.0
    for i, val in enumerate(metric_values):
        if val >= 1e6:
            label = f"{val:.2e}"
        elif val >= 1000:
            label = f"{val:,.0f}"
        elif val >= 1:
            label = f"{val:.2f}"
        else:
            label = f"{val:.4f}"
        ax_bar.text(
            val + max_val * 0.01, i, label,
            va="center", fontsize=8,
        )
    ax_bar.set_xlim(0, max_val * 1.20 if max_val > 0 else 1.0)

    # --- Right: kernel info card ---
    ax_info.axis("off")

    display_kernel = humanize_kernel_name(kernel_name)
    raw_short = kernel_name if len(kernel_name) <= 80 else kernel_name[:77] + "..."

    info_lines = [
        ("Cycle", str(cycle)),
        ("Trace window", f"{trace_s} s"),
        ("Kernel (short)", display_kernel),
        ("Total launches", f"{launches:,}"),
        ("Total GPU time", f"{total_ms:,.2f} ms"),
        ("Avg duration", f"{avg_us:,.2f} us"),
    ]

    y_start = 0.92
    y_step = 0.10
    for i, (key, val) in enumerate(info_lines):
        y = y_start - i * y_step
        ax_info.text(0.05, y, f"{key}:", fontsize=10, fontweight="bold",
                     transform=ax_info.transAxes, va="top")
        ax_info.text(0.48, y, val, fontsize=10,
                     transform=ax_info.transAxes, va="top")

    # Raw kernel name at the bottom, small font
    ax_info.text(
        0.05, 0.05, f"Raw: {raw_short}",
        fontsize=7, family="monospace", color="gray",
        transform=ax_info.transAxes, va="bottom",
        wrap=True,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def plot_profile_cycles_summary(json_paths, out_path: Path):
    """
    If multiple profiling cycles exist, generate a summary plot showing
    which kernel was profiled each cycle and how its metrics compare.
    """
    if len(json_paths) < 2:
        return  # Only useful with 2+ cycles

    cycles = []
    for p in json_paths:
        with p.open("r", encoding="utf-8") as f:
            cycles.append(json.load(f))

    # Collect all metric names across cycles
    all_metric_names = []
    for c in cycles:
        for m in c.get("metrics", {}):
            if m not in all_metric_names:
                all_metric_names.append(m)

    if not all_metric_names:
        return

    n_metrics = len(all_metric_names)
    n_cycles = len(cycles)

    fig, axes = plt.subplots(
        1, n_metrics,
        figsize=(4.5 * n_metrics, 5),
        squeeze=False,
    )

    cycle_labels = []
    for c in cycles:
        cnum = c.get("cycle", "?")
        kname = humanize_kernel_name(c.get("target_kernel", "?"))
        cycle_labels.append(f"C{cnum}\n{kname}")

    colors = plt.cm.tab10.colors

    for col, metric_name in enumerate(all_metric_names):
        ax = axes[0][col]
        vals = [c.get("metrics", {}).get(metric_name, 0.0) for c in cycles]
        bar_colors = [colors[i % len(colors)] for i in range(n_cycles)]
        bars = ax.bar(range(n_cycles), vals, color=bar_colors)
        ax.set_xticks(range(n_cycles))
        ax.set_xticklabels(cycle_labels, fontsize=8)
        ax.set_title(metric_name, fontsize=8, family="monospace")

        for bar, val in zip(bars, vals):
            if val >= 1e6:
                label = f"{val:.1e}"
            elif val >= 100:
                label = f"{val:,.0f}"
            else:
                label = f"{val:.2f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                label,
                ha="center", va="bottom", fontsize=7,
            )
        ax.set_ylim(0, max(vals) * 1.25 if max(vals) > 0 else 1.0)

    fig.suptitle("Profiling Metrics Across Cycles", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved plot to {out_path}")
    
def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Kernel hotspot plots (from tracer or profiler tracing phase) ---
    if GLOBAL_CSV.exists():
        plot_one(
            GLOBAL_CSV,
            PLOT_DIR / "kernel_hotspots_global.png",
            "Global Hotspots",
        )
    else:
        print(f"  Skipping global hotspots: {GLOBAL_CSV} not found")

    if RECENT_CSV.exists():
        plot_one(
            RECENT_CSV,
            PLOT_DIR / "kernel_hotspots_recent_5s.png",
            "Recent 5s Hotspots",
        )
    else:
        print(f"  Skipping recent hotspots: {RECENT_CSV} not found")

    # --- Two-way overhead comparison: plain vs tracer ---
    if OVERHEAD_PLAIN_JSON.exists() and OVERHEAD_TRACE_JSON.exists():
        plot_overhead(
            OVERHEAD_PLAIN_JSON,
            OVERHEAD_TRACE_JSON,
            PLOT_DIR / "overhead_compare.png",
        )
    else:
        print("  Skipping two-way overhead plot: missing llm_metrics.json or llm_trace_metrics.json")

    # --- Three-way overhead comparison: plain vs tracer vs profiler ---
    plot_overhead_three_way(
        OVERHEAD_PLAIN_JSON,
        OVERHEAD_TRACE_JSON,
        OVERHEAD_PROFILE_JSON,
        PLOT_DIR / "overhead_compare_three_way.png",
    )

    # --- Per-cycle profiling results plots ---
    cycle_jsons = _find_profile_cycle_jsons(OUT_DIR)
    if cycle_jsons:
        print(f"  Found {len(cycle_jsons)} profiling cycle(s)")
        for jp in cycle_jsons:
            cycle_num = jp.stem.replace("profile_cycle_", "")
            plot_profile_cycle(jp, PLOT_DIR / f"profile_cycle_{cycle_num}.png")

        # Summary comparison across cycles (only if 2+)
        plot_profile_cycles_summary(
            cycle_jsons, PLOT_DIR / "profile_cycles_summary.png"
        )
    else:
        print("  No profile_cycle_*.json files found; skipping profiling plots")


if __name__ == "__main__":
    main()