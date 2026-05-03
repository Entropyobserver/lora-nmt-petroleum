import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

RESULTS_FILE = Path("/crex/proj/uppmax2026-1-123/private/yaxj1/mt_oil_no/outputs/exp1_data_scaling/results.json")
OUTPUT_FILE  = Path("/crex/proj/uppmax2026-1-123/private/yaxj1/mt_oil_no/outputs/exp1_data_scaling/data_scaling_analysis.png")


def load_and_group(results_file):
    with open(results_file) as f:
        results = json.load(f)

    grouped = defaultdict(lambda: {"bleu": [], "chrf": []})
    for r in results:
        if r.get("failed"):
            continue
        size = r["data_size"]
        grouped[size]["bleu"].append(r["test_bleu"])
        grouped[size]["chrf"].append(r["test_chrf"])

    sizes = sorted(grouped.keys())
    bleus = [np.mean(grouped[s]["bleu"]) for s in sizes]
    chrfs = [np.mean(grouped[s]["chrf"]) for s in sizes]
    bleu_stds = [np.std(grouped[s]["bleu"]) for s in sizes]
    chrf_stds = [np.std(grouped[s]["chrf"]) for s in sizes]
    return sizes, bleus, chrfs, bleu_stds, chrf_stds


def compute_marginal_gains(sizes, bleus):
    gains = [0]
    for i in range(1, len(sizes)):
        gain = bleus[i] - bleus[i - 1]
        size_diff = sizes[i] - sizes[i - 1]
        gains.append(gain / (size_diff / 1000))
    return gains


def find_optimal(sizes, bleus, threshold=0.95):
    best = max(bleus)
    for i, bleu in enumerate(bleus):
        if bleu >= best * threshold:
            return i, best
    return None, best


def plot_quality(ax, sizes, bleus, chrfs, bleu_stds, chrf_stds, optimal_idx, best_bleu):
    color1 = "#2E86AB"
    color2 = "#A23B72"

    ax.errorbar(sizes, bleus, yerr=bleu_stds, fmt="o-", linewidth=2.5, markersize=8,
                color=color1, capsize=5, capthick=2, label="BLEU")
    ax.set_xlabel("Training Data Size", fontsize=12)
    ax.set_ylabel("BLEU Score", color=color1, fontsize=12)
    ax.tick_params(axis="y", labelcolor=color1)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=best_bleu * 0.95, color="red", linestyle="--", alpha=0.6,
               linewidth=1.5, label="95% BLEU threshold")

    if optimal_idx is not None:
        opt_size = sizes[optimal_idx]
        opt_bleu = bleus[optimal_idx]
        ax.plot(opt_size, opt_bleu, "r*", markersize=20, zorder=5)
        ax.annotate(
            f"Optimal Point\n{opt_size} samples\n{opt_bleu/best_bleu*100:.1f}% of max",
            xy=(opt_size, opt_bleu),
            xytext=(opt_size * 0.6, opt_bleu * 0.92),
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3", color="red", lw=2),
        )

    ax2 = ax.twinx()
    ax2.errorbar(sizes, chrfs, yerr=chrf_stds, fmt="s--", linewidth=2.5, markersize=8,
                 color=color2, capsize=5, capthick=2, label="chrF")
    ax2.set_ylabel("chrF Score", color=color2, fontsize=12)
    ax2.tick_params(axis="y", labelcolor=color2)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=10)
    ax.set_title("Translation Quality vs Training Data Size", fontsize=14, fontweight="bold")


def plot_marginal(ax, sizes, gains):
    from matplotlib.patches import Patch

    colors = []
    for g in gains:
        if g > 0.05: colors.append("#2E86AB")
        elif g > 0.01: colors.append("#4A9EBD")
        elif g > 0.005: colors.append("#F18F01")
        else: colors.append("#C73E1D")

    ax.bar(range(len(sizes)), gains, color=colors, alpha=0.7, edgecolor="black", linewidth=1.2)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels(sizes, rotation=45, ha="right")
    ax.set_xlabel("Training Data Size", fontsize=12)
    ax.set_ylabel("BLEU Gain per 1,000 Samples", fontsize=12)
    ax.set_title("Marginal Efficiency Analysis", fontsize=14, fontweight="bold")
    ax.axhline(y=0.005, color="red", linestyle="--", alpha=0.6, linewidth=1.5)
    ax.grid(True, alpha=0.3, axis="y")

    legend_elements = [
        Patch(facecolor="#2E86AB", alpha=0.7, edgecolor="black", label="High efficiency (>0.05)"),
        Patch(facecolor="#4A9EBD", alpha=0.7, edgecolor="black", label="Medium efficiency (0.01-0.05)"),
        Patch(facecolor="#F18F01", alpha=0.7, edgecolor="black", label="Low efficiency (0.005-0.01)"),
        Patch(facecolor="#C73E1D", alpha=0.7, edgecolor="black", label="Diminishing returns (<0.005)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)


def main():
    sizes, bleus, chrfs, bleu_stds, chrf_stds = load_and_group(RESULTS_FILE)
    gains = compute_marginal_gains(sizes, bleus)
    optimal_idx, best_bleu = find_optimal(sizes, bleus)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    plot_quality(ax1, sizes, bleus, chrfs, bleu_stds, chrf_stds, optimal_idx, best_bleu)
    plot_marginal(ax2, sizes, gains)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved to {OUTPUT_FILE}")
    print(f"Best BLEU: {best_bleu:.4f}, 95% threshold: {best_bleu*0.95:.4f}")
    if optimal_idx is not None:
        print(f"Optimal: {sizes[optimal_idx]} samples ({bleus[optimal_idx]:.4f} BLEU)")


if __name__ == "__main__":
    main()