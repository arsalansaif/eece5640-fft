#!/usr/bin/env python3
"""
Generate all figures for the EECE 5640 FFT benchmark write-up.

Usage:
    cd ~/eece5640-fft
    python3 scripts/plot_results.py --results-dir results --output-dir figures

Figures produced (PDF + PNG):
    fig1_throughput_synthetic   — GFlops vs FFT size, all 4 impls, synthetic
    fig2_speedup_synthetic      — Speedup over own_cpu vs FFT size
    fig3_transfer_breakdown     — H2D / kernel / D2H stacked bar (GPU only)
    fig4_batch_scaling          — GFlops vs batch size, all 4 impls
    fig5_thread_strong          — Strong scaling: speedup vs thread count
    fig6_thread_weak            — Weak scaling: time vs thread count
    fig7_roofline               — Roofline: own_gpu vs cuFFT on P100/V100
    fig8_speedup_oracle         — Speedup over own_cpu (ORACLE dataset)
    fig9_speedup_dns            — Speedup over own_cpu (DNS dataset)
    fig10_platform_compare      — P100 vs V100 throughput comparison
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Hardware specs ─────────────────────────────────────────────────────────────
HW = {
    "v100": {"peak_tflops": 14.0, "peak_bw_gbs": 900.0,
              "label": "V100-SXM2-32GB", "color": "#1f77b4"},
    "p100": {"peak_tflops":  9.3, "peak_bw_gbs": 549.0,
              "label": "P100-PCIe-12GB", "color": "#ff7f0e"},
}

# ── Colors and styles per implementation ──────────────────────────────────────
IMPL_STYLE = {
    "own_cpu": {"color": "#2ca02c", "linestyle": "-",  "marker": "o",
                "label": "own_cpu (Cooley-Tukey)"},
    "mkl":     {"color": "#98df8a", "linestyle": "--", "marker": "^",
                "label": "MKL (reference)"},
    "own_gpu": {"color": "#d62728", "linestyle": "-",  "marker": "s",
                "label": "own_gpu (CUDA kernel)"},
    "cufft":   {"color": "#1f77b4", "linestyle": "--", "marker": "D",
                "label": "cuFFT (reference)"},
}

PLATFORM_COLOR = {"p100": "#ff7f0e", "v100": "#1f77b4"}
PLATFORM_MARKER = {"p100": "o", "v100": "s"}

# ── Helpers ────────────────────────────────────────────────────────────────────
def load_csvs(results_dir: str) -> pd.DataFrame:
    files = list(set(
        glob.glob(os.path.join(results_dir, "*.csv")) +
        glob.glob(os.path.join(results_dir, "**/*.csv"), recursive=True)
    ))
    if not files:
        print(f"No CSV files found in {results_dir}", file=sys.stderr)
        sys.exit(1)
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            frames.append(df)
            print(f"  Loaded {f}  ({len(df)} rows)")
        except Exception as e:
            print(f"  Skipping {f}: {e}", file=sys.stderr)
    data = pd.concat(frames, ignore_index=True).drop_duplicates()
    return data


def savefig(fig, path: Path, dpi=200):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    print(f"  Saved {path}.pdf")
    plt.close(fig)


def pow2_formatter(x, _):
    if x > 0 and x == int(x) and (int(x) & (int(x) - 1)) == 0:
        return f"$2^{{{int(np.log2(x))}}}$"
    return f"{int(x)}"


# ── Fig 1: Throughput vs FFT size (synthetic) ─────────────────────────────────
def fig1_throughput(df, out_dir):
    sub = df[(df["dataset"] == "synthetic")].copy()
    platforms = [p for p in ["p100", "v100"] if p in sub["platform"].unique()]
    if sub.empty:
        return

    fig, axes = plt.subplots(1, len(platforms), figsize=(6*len(platforms), 5),
                              sharey=True, squeeze=False)
    fig.suptitle("Throughput vs FFT size — synthetic dataset", fontsize=13)

    for ax, plat in zip(axes[0], platforms):
        psub = sub[sub["platform"] == plat]
        for impl, style in IMPL_STYLE.items():
            grp = psub[psub["impl"] == impl].sort_values("fft_size")
            if grp.empty:
                continue
            ax.plot(grp["fft_size"], grp["gflops"],
                    label=style["label"], color=style["color"],
                    linestyle=style["linestyle"], marker=style["marker"],
                    markersize=5)

        # Mark own_gpu cliff at N=4096
        ax.axvline(4096, color="gray", linestyle=":", linewidth=1, alpha=0.7)
        ax.text(4096*1.1, ax.get_ylim()[0]*2 if ax.get_yscale()=="log" else 10,
                "shared→global\nmem switch", fontsize=7, color="gray")

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("FFT size (N)")
        ax.set_ylabel("GFlops/s")
        ax.set_title(HW.get(plat, {}).get("label", plat))
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(pow2_formatter))

    savefig(fig, out_dir / "fig1_throughput_synthetic")


# ── Fig 2: Speedup over own_cpu vs FFT size ───────────────────────────────────
def fig2_speedup(df, dataset, fig_name, out_dir, title_suffix=""):
    sub = df[df["dataset"] == dataset].copy()
    platforms = [p for p in ["p100", "v100"] if p in sub["platform"].unique()]
    if sub.empty:
        return

    fig, axes = plt.subplots(1, len(platforms), figsize=(6*len(platforms), 5),
                              sharey=True, squeeze=False)
    fig.suptitle(f"Speedup over own_cpu — {dataset} dataset{title_suffix}", fontsize=13)

    for ax, plat in zip(axes[0], platforms):
        psub = sub[sub["platform"] == plat]
        ref = psub[psub["impl"] == "own_cpu"][["fft_size", "kernel_ms"]].rename(
            columns={"kernel_ms": "ref_ms"})

        for impl in ["mkl", "own_gpu", "cufft"]:
            style = IMPL_STYLE[impl]
            grp = psub[psub["impl"] == impl][["fft_size", "kernel_ms",
                                               "total_wall_ms"]].copy()
            merged = pd.merge(ref, grp, on="fft_size").sort_values("fft_size")
            if merged.empty:
                continue

            # Kernel-only speedup
            ax.plot(merged["fft_size"], merged["ref_ms"] / merged["kernel_ms"],
                    label=f"{style['label']} (kernel)",
                    color=style["color"], linestyle=style["linestyle"],
                    marker=style["marker"], markersize=5)

            # Wall-time speedup (dashed, same color, thinner)
            if impl in ("own_gpu", "cufft"):
                ax.plot(merged["fft_size"],
                        merged["ref_ms"] / merged["total_wall_ms"],
                        color=style["color"], linestyle=":", linewidth=1,
                        alpha=0.6, label=f"{style['label']} (wall)")

        ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("FFT size (N)")
        ax.set_ylabel("Speedup vs own_cpu")
        ax.set_title(HW.get(plat, {}).get("label", plat))
        ax.legend(fontsize=7)
        ax.grid(True, which="both", alpha=0.3)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(pow2_formatter))

    savefig(fig, out_dir / fig_name)


# ── Fig 3: H2D / kernel / D2H stacked bar ─────────────────────────────────────
def fig3_transfer_breakdown(df, out_dir):
    sub = df[(df["dataset"] == "synthetic") &
             (df["impl"].isin(["own_gpu", "cufft"]))].copy()
    platforms = [p for p in ["p100", "v100"] if p in sub["platform"].unique()]
    if sub.empty:
        return

    fig, axes = plt.subplots(len(platforms), 2,
                              figsize=(12, 4*len(platforms)), squeeze=False)
    fig.suptitle("H2D / Kernel / D2H time breakdown (GPU, synthetic)", fontsize=13)

    for row, plat in enumerate(platforms):
        for col, impl in enumerate(["own_gpu", "cufft"]):
            ax = axes[row][col]
            grp = sub[(sub["platform"] == plat) &
                      (sub["impl"] == impl)].sort_values("fft_size")
            if grp.empty:
                ax.set_visible(False)
                continue

            x = np.arange(len(grp))
            ax.bar(x, grp["h2d_ms"],   label="H2D",    color="#aec7e8")
            ax.bar(x, grp["kernel_ms"], label="Kernel",
                   bottom=grp["h2d_ms"], color="#ff7f0e")
            ax.bar(x, grp["d2h_ms"],   label="D2H",
                   bottom=grp["h2d_ms"] + grp["kernel_ms"], color="#ffbb78")

            ax.set_xticks(x)
            ax.set_xticklabels([f"$2^{{{int(np.log2(n))}}}$"
                                 for n in grp["fft_size"]], fontsize=7, rotation=45)
            ax.set_xlabel("FFT size (N)")
            ax.set_ylabel("Time (ms)")
            ax.set_title(f"{impl.upper()} — {HW.get(plat,{}).get('label', plat)}")
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)

    savefig(fig, out_dir / "fig3_transfer_breakdown")


# ── Fig 4: Batch scaling ──────────────────────────────────────────────────────
def fig4_batch_scaling(df, out_dir):
    # Exp2 data: fft_size=1024 with varying batch
    sub = df[(df["fft_size"] == 1024) &
             (~df["dataset"].isin(["synthetic_strong", "synthetic_weak"]))].copy()
    platforms = [p for p in ["p100", "v100"] if p in sub["platform"].unique()]
    if sub.empty:
        return

    fig, axes = plt.subplots(1, len(platforms), figsize=(6*len(platforms), 5),
                              sharey=True, squeeze=False)
    fig.suptitle("Batch-depth scaling (FFT size = 1024, synthetic)", fontsize=13)

    for ax, plat in zip(axes[0], platforms):
        psub = sub[sub["platform"] == plat]
        for impl, style in IMPL_STYLE.items():
            grp = psub[psub["impl"] == impl].sort_values("batch_size")
            if grp.empty:
                continue
            ax.plot(grp["batch_size"], grp["gflops"],
                    label=style["label"], color=style["color"],
                    linestyle=style["linestyle"], marker=style["marker"],
                    markersize=4)

        ax.set_xscale("log", base=2)
        ax.set_xlabel("Batch size")
        ax.set_ylabel("GFlops/s (kernel only)")
        ax.set_title(HW.get(plat, {}).get("label", plat))
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

    savefig(fig, out_dir / "fig4_batch_scaling")


# ── Fig 5: Thread scaling — strong ────────────────────────────────────────────
def fig5_thread_strong(df, out_dir):
    sub = df[df["dataset"] == "synthetic_strong"].copy()
    if sub.empty:
        print("  No strong-scaling data, skipping fig5")
        return

    platforms = [p for p in ["p100", "v100"] if p in sub["platform"].unique()]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title("Strong scaling — fixed total work, vary threads (N=1024)", fontsize=12)

    for plat in platforms:
        for impl in ["own_cpu", "mkl"]:
            style = IMPL_STYLE[impl]
            grp = sub[(sub["platform"] == plat) &
                      (sub["impl"] == impl)].sort_values("num_threads")
            if grp.empty:
                continue
            # Speedup relative to 1-thread run
            t1 = grp[grp["num_threads"] == 1]["kernel_ms"].values
            if len(t1) == 0:
                continue
            speedup = t1[0] / grp["kernel_ms"].values
            label = f"{style['label']} ({HW.get(plat,{}).get('label', plat)})"
            ax.plot(grp["num_threads"], speedup,
                    color=style["color"], linestyle=style["linestyle"],
                    marker=style["marker"], markersize=6, label=label)

    # Ideal speedup
    threads = np.array([1, 2, 4, 8, 16])
    ax.plot(threads, threads, "k--", linewidth=1, alpha=0.5, label="Ideal (linear)")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Thread count")
    ax.set_ylabel("Speedup (vs 1 thread)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: str(int(x)) if x >= 1 else f"{x:.1f}"))
    savefig(fig, out_dir / "fig5_thread_strong")


# ── Fig 6: Thread scaling — weak ──────────────────────────────────────────────
def fig6_thread_weak(df, out_dir):
    sub = df[df["dataset"] == "synthetic_weak"].copy()
    if sub.empty:
        print("  No weak-scaling data, skipping fig6")
        return

    platforms = [p for p in ["p100", "v100"] if p in sub["platform"].unique()]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title("Weak scaling — work per thread fixed, vary threads (N=1024)", fontsize=12)

    for plat in platforms:
        for impl in ["own_cpu", "mkl"]:
            style = IMPL_STYLE[impl]
            grp = sub[(sub["platform"] == plat) &
                      (sub["impl"] == impl)].sort_values("num_threads")
            if grp.empty:
                continue
            label = f"{style['label']} ({HW.get(plat,{}).get('label', plat)})"
            ax.plot(grp["num_threads"], grp["kernel_ms"],
                    color=style["color"], linestyle=style["linestyle"],
                    marker=style["marker"], markersize=6, label=label)

    # Ideal: flat line (time at 1 thread)
    for plat in platforms:
        for impl in ["own_cpu", "mkl"]:
            grp = sub[(sub["platform"] == plat) &
                      (sub["impl"] == impl)].sort_values("num_threads")
            if grp.empty:
                continue
            t1 = grp[grp["num_threads"] == 1]["kernel_ms"].values
            if len(t1) == 0:
                continue
            ax.axhline(t1[0], color=IMPL_STYLE[impl]["color"],
                       linestyle=":", alpha=0.4, linewidth=1)
            break
        break

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Thread count")
    ax.set_ylabel("Kernel time (ms)  [ideal: flat]")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: str(int(x)) if x >= 1 else f"{x:.1f}"))
    savefig(fig, out_dir / "fig6_thread_weak")


# ── Fig 7: Roofline ───────────────────────────────────────────────────────────
def fig7_roofline(df, out_dir):
    """
    Arithmetic intensity for FFT: I = 5*log2(N)/16  [FLOP/byte]
    (numerator: 5*N*log2(N) FLOPs; denominator: 2*N*8 bytes read+write)
    """
    sub = df[(df["dataset"] == "synthetic") &
             (df["impl"].isin(["own_gpu", "cufft"]))].copy()
    if sub.empty:
        return

    sub["arith_intensity"] = 5.0 * np.log2(sub["fft_size"].astype(float)) / 16.0

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title("Roofline model — GPU FFT kernels (synthetic)", fontsize=12)

    # Roofline ceilings
    ai_range = np.logspace(-1, 2, 300)
    for plat, hw in HW.items():
        if plat not in sub["platform"].unique():
            continue
        peak_f = hw["peak_tflops"] * 1e3
        peak_b = hw["peak_bw_gbs"]
        roof   = np.minimum(peak_b * ai_range, peak_f)
        ax.plot(ai_range, roof, color=hw["color"], linestyle="--",
                linewidth=1.5, alpha=0.7,
                label=f"Roofline {hw['label']} (peak {hw['peak_tflops']} TF, {peak_b} GB/s)")

    # Achieved points
    for (impl, plat), grp in sub.groupby(["impl", "platform"]):
        hw = HW.get(plat, {})
        style = IMPL_STYLE[impl]
        grp = grp.sort_values("arith_intensity")
        ax.scatter(grp["arith_intensity"], grp["gflops"],
                   color=style["color"],
                   marker=PLATFORM_MARKER.get(plat, "o"),
                   s=40, zorder=5, alpha=0.85,
                   label=f"{style['label']} — {hw.get('label', plat)}")

        # Annotate a few N values
        for _, row in grp.iterrows():
            n = int(row["fft_size"])
            if n in (128, 1024, 4096, 65536, 1048576):
                ax.annotate(f"N={n}", (row["arith_intensity"], row["gflops"]),
                            fontsize=6, textcoords="offset points", xytext=(4, 2))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("Achieved GFlops/s")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    savefig(fig, out_dir / "fig7_roofline")


# ── Fig 8/9: Speedup on ORACLE and DNS ────────────────────────────────────────
# (reuses fig2_speedup with different dataset)


# ── Fig 10: Platform comparison P100 vs V100 ─────────────────────────────────
def fig10_platform_compare(df, out_dir):
    sub = df[(df["dataset"] == "synthetic") &
             (df["impl"].isin(["own_gpu", "cufft"]))].copy()

    platforms = [p for p in ["p100", "v100"] if p in sub["platform"].unique()]
    if len(platforms) < 2:
        print("  Only one platform available, skipping fig10")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("P100 vs V100: kernel throughput (synthetic)", fontsize=13)

    for ax, impl in zip(axes, ["own_gpu", "cufft"]):
        style = IMPL_STYLE[impl]
        for plat in platforms:
            grp = sub[(sub["platform"] == plat) &
                      (sub["impl"] == impl)].sort_values("fft_size")
            hw  = HW.get(plat, {"label": plat, "color": "gray"})
            ax.plot(grp["fft_size"], grp["gflops"],
                    label=hw["label"], color=hw["color"],
                    marker=PLATFORM_MARKER.get(plat, "o"), markersize=5)

        ax.set_xscale("log", base=2)
        ax.set_xlabel("FFT size (N)")
        ax.set_ylabel("GFlops/s")
        ax.set_title(style["label"])
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(pow2_formatter))

    savefig(fig, out_dir / "fig10_platform_compare")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--output-dir",  default="figures")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    print(f"Loading CSVs from: {args.results_dir}")
    df = load_csvs(args.results_dir)
    print(f"Total rows: {len(df)}")
    print(f"Platforms:  {sorted(df['platform'].unique())}")
    print(f"Datasets:   {sorted(df['dataset'].unique())}")
    print(f"Impls:      {sorted(df['impl'].unique())}")

    print("\nGenerating figures...")
    fig1_throughput(df, out_dir)
    fig2_speedup(df, "synthetic", "fig2_speedup_synthetic", out_dir)
    fig3_transfer_breakdown(df, out_dir)
    fig4_batch_scaling(df, out_dir)
    fig5_thread_strong(df, out_dir)
    fig6_thread_weak(df, out_dir)
    fig7_roofline(df, out_dir)
    fig2_speedup(df, "oracle",    "fig8_speedup_oracle", out_dir, " (ORACLE RF)")
    fig2_speedup(df, "dns",       "fig9_speedup_dns",    out_dir, " (DNS Challenge)")
    fig10_platform_compare(df, out_dir)

    print(f"\nDone. Figures saved to: {out_dir}/")


if __name__ == "__main__":
    main()
