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

    fig.subplots_adjust(bottom=0.42, wspace=0.35)
    fig.text(
        0.01,
        0.02,
        footnote_text,
        ha="left",
        va="bottom",
        fontsize=15,
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


def _fmt_value(val):
    """Format a metric value for display on a bar label."""
    if val >= 1e9:
        return f"{val:.2e}"
    elif val >= 1e6:
        return f"{val / 1e6:,.1f}M"
    elif val >= 1e3:
        return f"{val / 1e3:,.1f}K"
    elif val >= 1:
        return f"{val:,.2f}"
    else:
        return f"{val:.4f}"


def _bar_label(ax, bars, values, fontsize=8):
    """Add value annotations on top of bars."""
    max_val = max(values) if values else 1.0
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_val * 0.01,
            _fmt_value(val),
            ha="center", va="bottom", fontsize=fontsize,
        )


def _hbar_label(ax, bars, values, fontsize=8):
    """Add value annotations to the right of horizontal bars."""
    max_val = max(abs(v) for v in values) if values else 1.0
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + max_val * 0.02, bar.get_y() + bar.get_height() / 2,
            _fmt_value(val),
            va="center", fontsize=fontsize,
        )


def plot_profile_cycle(json_path: Path, out_path: Path):
    """
    Generate a comprehensive multi-subplot figure for a single profiling
    cycle.  The layout adapts based on which metrics are present:

      Row 1:  [Raw metric values (hbar)]  [Kernel info card]
      Row 2:  [SM utilization gauge]  [Memory throughput]  [Bottleneck diagnosis]

    Row 2 panels are only drawn when the relevant default metrics are
    available (cycles_elapsed, cycles_active, warps_active, dram bytes).
    """
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    cycle = data.get("cycle", "?")
    kernel_name = data.get("target_kernel", "unknown")
    trace_s = data.get("trace_duration_s", "?")
    launches = data.get("total_launches", 0)
    total_ms = data.get("total_gpu_time_ms", 0.0)
    avg_us = data.get("avg_duration_us", 0.0)
    num_passes = data.get("num_replay_passes", 0)
    profile_wall_us = data.get("profile_wall_time_us", 0.0)
    profiled_kernel_us = data.get("profiled_kernel_time_us", 0.0)
    metrics = data.get("metrics", {})

    if not metrics:
        print(f"  Skipping {json_path.name}: no metrics collected")
        return

    # ------------------------------------------------------------------
    # Derive higher-level insights from the raw metrics
    # ------------------------------------------------------------------
    cycles_elapsed = metrics.get("sm__cycles_elapsed.avg", 0.0)
    cycles_active = metrics.get("sm__cycles_active.avg", 0.0)
    warps_active = metrics.get("sm__warps_active.avg", 0.0)
    dram_read = metrics.get("dram__bytes_read.sum", 0.0)
    dram_write = metrics.get("dram__bytes_write.sum", 0.0)

    has_sm = cycles_elapsed > 0
    has_warps = warps_active > 0 or "sm__warps_active.avg" in metrics
    has_dram = "dram__bytes_read.sum" in metrics or "dram__bytes_write.sum" in metrics

    active_ratio = (cycles_active / cycles_elapsed) if has_sm else None
    dram_total = dram_read + dram_write

    # ------------------------------------------------------------------
    # Build the figure layout
    # ------------------------------------------------------------------
    # Row 1 (counters + info card) is compact; row 2 (analysis) is taller
    # so the SM / memory / diagnosis panels have room for labels and legends.
    has_row2 = has_sm or has_dram  # whether we need the analysis row

    if has_row2:
        fig = plt.figure(figsize=(25, 13))
        gs = fig.add_gridspec(2, 3, height_ratios=[2, 3], hspace=0.38, wspace=0.35)
        ax_raw = fig.add_subplot(gs[0, 0:2])   # raw metrics bar chart (wide)
        ax_info = fig.add_subplot(gs[0, 2])     # kernel info card
        ax_sm = fig.add_subplot(gs[1, 0])       # SM utilization
        ax_mem = fig.add_subplot(gs[1, 1])       # memory breakdown
        ax_diag = fig.add_subplot(gs[1, 2])      # bottleneck diagnosis
    else:
        fig = plt.figure(figsize=(16, 5.5))
        gs = fig.add_gridspec(1, 3, width_ratios=[2, 2, 1])
        ax_raw = fig.add_subplot(gs[0, 0:2])
        ax_info = fig.add_subplot(gs[0, 2])
        ax_sm = ax_mem = ax_diag = None

    # ------------------------------------------------------------------
    # Panel 1: Raw metric values (horizontal bar chart)
    # ------------------------------------------------------------------
    metric_names = list(metrics.keys())
    metric_values = [metrics[k] for k in metric_names]
    short_names = [m if len(m) <= 40 else m[:37] + "..." for m in metric_names]

    colors_raw = []
    for m in metric_names:
        if "dram" in m:
            colors_raw.append("#c44e52")
        elif "warp" in m or "occupancy" in m:
            colors_raw.append("#dd8452")
        elif "lts" in m or "l1tex" in m:
            colors_raw.append("#8c564b")
        else:
            colors_raw.append("#4c72b0")

    y_pos = list(range(len(short_names)))
    bars_raw = ax_raw.barh(y_pos, metric_values, color=colors_raw)
    ax_raw.set_yticks(y_pos)
    ax_raw.set_yticklabels(short_names, fontsize=9, family="monospace")
    ax_raw.set_xlabel("Value")
    ax_raw.set_title("Collected Hardware Counters", fontsize=11, fontweight="bold")
    ax_raw.invert_yaxis()
    _hbar_label(ax_raw, bars_raw, metric_values, fontsize=8)
    max_raw = max(metric_values) if metric_values else 1.0
    ax_raw.set_xlim(0, max_raw * 1.25 if max_raw > 0 else 1.0)

    # ------------------------------------------------------------------
    # Panel 2: Kernel info card
    # ------------------------------------------------------------------
    ax_info.axis("off")
    display_kernel = humanize_kernel_name(kernel_name)
    raw_short = kernel_name if len(kernel_name) <= 70 else kernel_name[:67] + "..."

    card_lines = [
        ("Cycle", str(cycle)),
        ("Trace window", f"{trace_s} s"),
        ("Kernel", display_kernel),
        ("Launches", f"{launches:,}"),
        ("Total GPU time", f"{total_ms:,.2f} ms"),
        ("Avg duration", f"{avg_us:,.2f} us"),
    ]
    if num_passes > 0:
        card_lines.append(("Total runs (with replay)", str(num_passes)))
    if profile_wall_us > 0:
        card_lines.append(("Profile wall time", f"{profile_wall_us:,.1f} us"))
    if profiled_kernel_us > 0:
        card_lines.append(("Profiled kernel time", f"{profiled_kernel_us:,.1f} us"))
    if active_ratio is not None:
        card_lines.append(("SM active ratio", f"{active_ratio:.1%}"))
    if has_dram:
        card_lines.append(("DRAM total", _fmt_value(dram_total) + " B"))

    y_start = 0.95
    y_step = 0.075
    for i, (key, val) in enumerate(card_lines):
        y = y_start - i * y_step
        ax_info.text(0.02, y, f"{key}:", fontsize=12, fontweight="bold",
                     transform=ax_info.transAxes, va="top")
        ax_info.text(0.45, y, val, fontsize=12,
                     transform=ax_info.transAxes, va="top")

    ax_info.text(
        0.02, 0.03, f"Raw: {raw_short}",
        fontsize=8, family="monospace", color="gray",
        transform=ax_info.transAxes, va="bottom", wrap=True,
    )
    ax_info.set_title("Kernel Identity", fontsize=13, fontweight="bold")

    # ------------------------------------------------------------------
    # Panel 3: SM utilization breakdown (stacked bar)
    # ------------------------------------------------------------------
    if ax_sm is not None and has_sm:
        active = cycles_active
        idle = max(0, cycles_elapsed - cycles_active)
        bars_a = ax_sm.bar(["SM Cycles"], [active], color="#4c72b0", label="Active")
        ax_sm.bar(["SM Cycles"], [idle], bottom=[active], color="#d9d9d9", label="Idle")
        ax_sm.set_ylabel("Cycles (avg per SM)")
        ax_sm.set_title("SM Utilization", fontsize=13, fontweight="bold")

        ratio_pct = (active_ratio * 100) if active_ratio is not None else 0
        # Extra headroom for label + legend above bars
        ax_sm.set_ylim(0, cycles_elapsed * 1.35 if cycles_elapsed > 0 else 1.0)
        ax_sm.text(
            0, active + idle + (cycles_elapsed * 0.02),
            f"{ratio_pct:.1f}% active",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
            color="#4c72b0",
        )

        if has_warps:
            ax_sm2 = ax_sm.twinx()
            ax_sm2.bar(["Warps Active"], [warps_active], color="#dd8452", width=0.4,
                       label="Warps active")
            ax_sm2.set_ylabel("Warps active (avg)", color="#dd8452")
            ax_sm2.tick_params(axis="y", labelcolor="#dd8452")
            ax_sm2.set_ylim(0, warps_active * 2.0 if warps_active > 0 else 1.0)
            ax_sm2.text(
                1, warps_active + warps_active * 0.05,
                f"{warps_active:.1f}",
                ha="center", va="bottom", fontsize=12, color="#dd8452",
            )
            # Combine legends from both axes, place above the plot
            handles1, labels1 = ax_sm.get_legend_handles_labels()
            handles2, labels2 = ax_sm2.get_legend_handles_labels()
            ax_sm.legend(handles1 + handles2, labels1 + labels2,
                         loc="upper center", fontsize=12, ncol=3,
                         bbox_to_anchor=(0.5, 1.0))
        else:
            ax_sm.legend(loc="upper center", fontsize=12, ncol=2,
                         bbox_to_anchor=(0.5, 1.0))
    elif ax_sm is not None:
        ax_sm.axis("off")
        ax_sm.text(0.5, 0.5, "SM metrics\nnot collected",
                   ha="center", va="center", fontsize=12, color="gray",
                   transform=ax_sm.transAxes)

    # ------------------------------------------------------------------
    # Panel 4: DRAM memory throughput breakdown
    # ------------------------------------------------------------------
    if ax_mem is not None and has_dram:
        mem_labels = []
        mem_vals = []
        mem_colors = []
        if "dram__bytes_read.sum" in metrics:
            mem_labels.append("DRAM Read")
            mem_vals.append(dram_read)
            mem_colors.append("#4c72b0")
        if "dram__bytes_write.sum" in metrics:
            mem_labels.append("DRAM Write")
            mem_vals.append(dram_write)
            mem_colors.append("#c44e52")
        if len(mem_vals) == 2:
            mem_labels.append("Total")
            mem_vals.append(dram_total)
            mem_colors.append("#2ca02c")

        bars_mem = ax_mem.bar(mem_labels, mem_vals, color=mem_colors)
        ax_mem.set_ylabel("Bytes")
        ax_mem.set_title("DRAM Traffic", fontsize=11, fontweight="bold")
        _bar_label(ax_mem, bars_mem, mem_vals, fontsize=9)
        ax_mem.set_ylim(0, max(mem_vals) * 1.35 if max(mem_vals) > 0 else 1.0)

        # Add read/write ratio annotation
        if dram_total > 0 and len(mem_vals) >= 2:
            rd_pct = dram_read / dram_total * 100
            wr_pct = dram_write / dram_total * 100
            ax_mem.text(
                0.5, 0.92,
                f"Read {rd_pct:.0f}% / Write {wr_pct:.0f}%",
                ha="center", va="top", fontsize=12,
                transform=ax_mem.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8),
            )
    elif ax_mem is not None:
        ax_mem.axis("off")
        ax_mem.text(0.5, 0.5, "DRAM metrics\nnot collected",
                   ha="center", va="center", fontsize=12, color="gray",
                   transform=ax_mem.transAxes)

    # ------------------------------------------------------------------
    # Panel 5: Bottleneck diagnosis
    # ------------------------------------------------------------------
    if ax_diag is not None:
        ax_diag.axis("off")
        ax_diag.set_title("Bottleneck Analysis", fontsize=13, fontweight="bold")

        diag_lines = []
        diag_color = "black"

        if has_sm and has_dram:
            if active_ratio is not None and active_ratio > 0.8 and dram_total < 1e6:
                diag_lines.append("COMPUTE-BOUND")
                diag_lines.append("SMs are busy; memory is not the bottleneck.")
                diag_lines.append("Consider algorithmic optimizations.")
                diag_color = "#4c72b0"
            elif active_ratio is not None and active_ratio < 0.5 and dram_total > 1e6:
                diag_lines.append("MEMORY-BOUND")
                diag_lines.append("SMs are often stalled waiting for DRAM.")
                diag_lines.append("Consider reducing memory traffic,")
                diag_lines.append("improving data locality, or using")
                diag_lines.append("shared memory / tiling.")
                diag_color = "#c44e52"
            elif active_ratio is not None and active_ratio < 0.5 and dram_total < 1e6:
                diag_lines.append("LATENCY-BOUND")
                diag_lines.append("SMs are idle but DRAM traffic is low.")
                diag_lines.append("Likely stalled on sync, small grid,")
                diag_lines.append("or L2 misses not reaching DRAM.")
                diag_color = "#dd8452"
            else:
                diag_lines.append("MIXED / BALANCED")
                diag_lines.append("No single dominant bottleneck detected.")
                diag_lines.append("Profile with more metrics for deeper")
                diag_lines.append("analysis.")
                diag_color = "#2ca02c"

            if has_warps and warps_active < 8:
                diag_lines.append("")
                diag_lines.append("LOW OCCUPANCY")
                diag_lines.append(f"Only {warps_active:.1f} warps active (avg).")
                diag_lines.append("Consider reducing register/shmem usage")
                diag_lines.append("or increasing grid size.")
        elif has_sm:
            if active_ratio is not None:
                label = "HIGH" if active_ratio > 0.7 else "LOW"
                diag_lines.append(f"SM Active Ratio: {label}")
                diag_lines.append(f"({active_ratio:.1%} of elapsed cycles)")
                diag_color = "#4c72b0" if active_ratio > 0.7 else "#dd8452"
            else:
                diag_lines.append("Insufficient data for diagnosis.")
        else:
            diag_lines.append("Need sm__cycles_elapsed.avg and")
            diag_lines.append("dram__bytes_*.sum for bottleneck")
            diag_lines.append("diagnosis. Set INJECTION_METRICS.")

        # Render diagnosis text
        y = 0.85
        for i, line in enumerate(diag_lines):
            weight = "bold" if i == 0 else "normal"
            color = diag_color if i == 0 else "black"
            fontsize = 12
            ax_diag.text(
                0.05, y, line,
                fontsize=fontsize, fontweight=weight, color=color,
                transform=ax_diag.transAxes, va="top",
            )
            y -= 0.10 if i == 0 else 0.08

    fig.suptitle(
        f"Profiling Results \u2014 Cycle {cycle}",
        fontsize=14, fontweight="bold", y=0.98,
    )
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
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


def plot_profile_cycle_pair(c1_path: Path, c2_path: Path, out_path: Path):
    """
    Side-by-side comparison of two profiling cycles (typically cycle 1 and
    cycle 2). Shows the two kernels' identity cards, their SM utilization
    breakdown, and their DRAM traffic breakdown in one figure so the
    per-cycle hot-kernel behavior can be contrasted at a glance.
    """
    with c1_path.open("r", encoding="utf-8") as f:
        c1 = json.load(f)
    with c2_path.open("r", encoding="utf-8") as f:
        c2 = json.load(f)

    cycles = [c1, c2]
    labels = []
    for c in cycles:
        cnum = c.get("cycle", "?")
        kname = humanize_kernel_name(c.get("target_kernel", "?"))
        labels.append(f"Cycle {cnum}\n{kname}")

    # Extract derived quantities per cycle
    def _derive(c):
        m = c.get("metrics", {}) or {}
        elapsed = m.get("sm__cycles_elapsed.avg", 0.0)
        active = m.get("sm__cycles_active.avg", 0.0)
        idle = max(0.0, elapsed - active)
        warps = m.get("sm__warps_active.avg", 0.0)
        rd = m.get("dram__bytes_read.sum", 0.0)
        wr = m.get("dram__bytes_write.sum", 0.0)
        ratio = (active / elapsed) if elapsed > 0 else 0.0
        return {
            "elapsed": elapsed,
            "active": active,
            "idle": idle,
            "ratio": ratio,
            "warps": warps,
            "dram_read": rd,
            "dram_write": wr,
            "dram_total": rd + wr,
        }

    d = [_derive(c) for c in cycles]

    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.0], wspace=0.35)
    ax_info = fig.add_subplot(gs[0, 0])
    ax_sm = fig.add_subplot(gs[0, 1])
    ax_mem = fig.add_subplot(gs[0, 2])

    # ---------------- Identity card ----------------
    ax_info.axis("off")
    ax_info.set_title("Kernel Identity", fontsize=13, fontweight="bold")
    y = 0.97
    for i, c in enumerate(cycles):
        cnum = c.get("cycle", "?")
        kname = humanize_kernel_name(c.get("target_kernel", "?"))
        launches = c.get("total_launches", 0)
        avg_us = c.get("avg_duration_us", 0.0)
        num_passes = c.get("num_replay_passes", 0)
        profile_wall_us = c.get("profile_wall_time_us", 0.0)
        profiled_kernel_us = c.get("profiled_kernel_time_us", 0.0)
        color = "#4c72b0" if i == 0 else "#dd8452"
        ax_info.text(0.02, y, f"Cycle {cnum}", fontsize=12, fontweight="bold",
                     color=color, transform=ax_info.transAxes, va="top")
        y -= 0.055
        ax_info.text(0.02, y, kname, fontsize=10,
                     transform=ax_info.transAxes, va="top")
        y -= 0.048
        ax_info.text(0.02, y, f"launches: {launches:,}", fontsize=9,
                     transform=ax_info.transAxes, va="top")
        y -= 0.040
        ax_info.text(0.02, y, f"trace avg:      {avg_us:.1f} us", fontsize=9,
                     family="monospace",
                     transform=ax_info.transAxes, va="top")
        y -= 0.040
        if num_passes > 0:
            ax_info.text(0.02, y,
                         f"Total runs (original run + replay):  {num_passes}",
                         fontsize=9, family="monospace",
                         transform=ax_info.transAxes, va="top")
            y -= 0.040
        if profile_wall_us > 0:
            ax_info.text(0.02, y,
                         f"profile wall:   {profile_wall_us:.1f} us",
                         fontsize=9, family="monospace",
                         transform=ax_info.transAxes, va="top")
            y -= 0.040
        if profiled_kernel_us > 0:
            ax_info.text(0.02, y,
                         f"per-pass kernel:{profiled_kernel_us:.1f} us",
                         fontsize=9, family="monospace",
                         transform=ax_info.transAxes, va="top")
            y -= 0.040
        ax_info.text(0.02, y,
                     f"SM active:      {d[i]['ratio'] * 100:.1f}%",
                     fontsize=9, family="monospace",
                     transform=ax_info.transAxes, va="top")
        y -= 0.065

    # ---------------- SM utilization ----------------
    x = [0, 1]
    actives = [d[0]["active"], d[1]["active"]]
    idles = [d[0]["idle"], d[1]["idle"]]
    elapseds = [d[0]["elapsed"], d[1]["elapsed"]]
    ax_sm.bar(x, actives, color="#4c72b0", label="Active")
    ax_sm.bar(x, idles, bottom=actives, color="#d9d9d9", label="Idle")
    ax_sm.set_xticks(x)
    ax_sm.set_xticklabels(labels, fontsize=10)
    ax_sm.set_ylabel("Cycles (avg per SM)")
    ax_sm.set_title("SM Utilization", fontsize=13, fontweight="bold")
    ymax = max(elapseds) if max(elapseds) > 0 else 1.0
    ax_sm.set_ylim(0, ymax * 1.30)
    for xi, di in zip(x, d):
        if di["elapsed"] > 0:
            ax_sm.text(
                xi, di["elapsed"] + ymax * 0.02,
                f"{di['ratio'] * 100:.1f}% active",
                ha="center", va="bottom", fontsize=11,
                fontweight="bold", color="#4c72b0",
            )
    ax_sm.legend(loc="upper center", fontsize=10, ncol=2,
                 bbox_to_anchor=(0.5, 1.0))

    # ---------------- DRAM traffic ----------------
    reads = [d[0]["dram_read"], d[1]["dram_read"]]
    writes = [d[0]["dram_write"], d[1]["dram_write"]]
    totals = [d[0]["dram_total"], d[1]["dram_total"]]
    ax_mem.bar(x, reads, color="#55a868", label="Read")
    ax_mem.bar(x, writes, bottom=reads, color="#c44e52", label="Write")
    ax_mem.set_xticks(x)
    ax_mem.set_xticklabels(labels, fontsize=10)
    ax_mem.set_ylabel("DRAM bytes")
    ax_mem.set_title("DRAM Traffic", fontsize=13, fontweight="bold")
    mmax = max(totals) if max(totals) > 0 else 1.0
    ax_mem.set_ylim(0, mmax * 1.30)
    for xi, di in zip(x, d):
        if di["dram_total"] > 0:
            ax_mem.text(
                xi, di["dram_total"] + mmax * 0.02,
                _fmt_value(di["dram_total"]) + " B",
                ha="center", va="bottom", fontsize=11,
                fontweight="bold",
            )
    ax_mem.legend(loc="upper center", fontsize=10, ncol=2,
                  bbox_to_anchor=(0.5, 1.0))

    fig.suptitle(
        f"Profiling Cycle {c1.get('cycle', '?')} vs Cycle {c2.get('cycle', '?')}",
        fontsize=14, fontweight="bold", y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
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

        # Focused side-by-side comparison of cycle 1 and cycle 2
        if len(cycle_jsons) >= 2:
            plot_profile_cycle_pair(
                cycle_jsons[0],
                cycle_jsons[1],
                PLOT_DIR / "profile_cycle_1_vs_2.png",
            )
    else:
        print("  No profile_cycle_*.json files found; skipping profiling plots")


if __name__ == "__main__":
    main()