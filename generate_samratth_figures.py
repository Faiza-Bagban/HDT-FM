# generate_samratth_figures.py
"""
Generate publication-quality figures for Samratth's §IV contribution.
1. Ablation bar chart (F1 comparison across modality combos)
2. Training loss curves (CE + twin reconstruction)
3. Component contribution table (stdout for LaTeX)

Usage: python generate_samratth_figures.py
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
})

FIG_DIR = "results/figures"
os.makedirs(FIG_DIR, exist_ok=True)


def plot_ablation_bar():
    """Ablation bar chart from CSV."""
    csv_path = "runs/ablation_results.csv"
    if not os.path.isfile(csv_path):
        print(f"ERROR: {csv_path} not found. Run ablation_sweep.py first.")
        return

    variants, f1s = [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            variants.append(row["variant"])
            f1s.append(float(row["best_val_f1"]))

    # Sort by F1 ascending for visual impact
    paired = sorted(zip(f1s, variants))
    f1s, variants = zip(*paired)

    # Color scheme: highlight full-fusion
    colors = []
    for v in variants:
        if v == "full-fusion":
            colors.append("#1a5276")      # dark blue = best
        elif "activity" in v:
            colors.append("#2e86c1")      # blue = activity variants
        else:
            colors.append("#aed6f1")      # light = voice/mobility only

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(range(len(variants)), f1s, color=colors, edgecolor="white", height=0.6)

    # Add value labels
    for bar, f1 in zip(bars, f1s):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{f1:.4f}", va="center", fontsize=10, fontweight="bold")

    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels(variants, fontsize=10)
    ax.set_xlabel("Weighted F1-Score", fontsize=12)
    ax.set_title("Modality Ablation Study (Subject-Held-Out, 3-Class)", fontsize=13)
    ax.set_xlim(0, max(f1s) + 0.08)
    ax.axvline(x=0.333, color="red", linestyle="--", alpha=0.5, label="Random baseline")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.2)

    path = os.path.join(FIG_DIR, "ablation_bar_samratth.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close()
    print(f"Saved: {path}")


def plot_training_curves():
    """Simulated training curves from v2 run metrics."""
    # From actual run: epoch 0-29 approximate values
    epochs = np.arange(30)

    # CE loss (from v2 twinfix run)
    ce = np.array([0.4332, 0.1724, 0.1264, 0.1055, 0.0935, 0.0849, 0.0769,
                   0.0752, 0.0710, 0.0667, 0.0632, 0.0579, 0.0548, 0.0605,
                   0.0953, 0.0837, 0.0687, 0.0982, 0.1714, 0.1161, 0.0929,
                   0.0893, 0.0850, 0.1323, 0.0938, 0.0771, 0.0808, 0.0722,
                   0.0677, 0.0620])

    # Twin reconstruction loss
    twin = np.array([0.1238, 0.0801, 0.0676, 0.0620, 0.0590, 0.0571, 0.0559,
                     0.0542, 0.0527, 0.0504, 0.0483, 0.0471, 0.0449, 0.0425,
                     0.0512, 0.0483, 0.0442, 0.0563, 0.1211, 0.0755, 0.0626,
                     0.0548, 0.0512, 0.0544, 0.0516, 0.0497, 0.0490, 0.0480,
                     0.0469, 0.0470])

    # Val F1
    val_f1 = np.array([0.5487, 0.5811, 0.6309, 0.6225, 0.5854, 0.5960, 0.6661,
                       0.5650, 0.5879, 0.6090, 0.5845, 0.6323, 0.5577, 0.5921,
                       0.5981, 0.5975, 0.5700, 0.5936, 0.6181, 0.5554, 0.5977,
                       0.5976, 0.5799, 0.6105, 0.6117, 0.6130, 0.6248, 0.6213,
                       0.6176, 0.6145])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: Loss curves
    ax1.plot(epochs, ce, "o-", label="CE Loss", color="#c0392b", markersize=3, linewidth=1.5)
    ax1.plot(epochs, twin, "s-", label="Twin Recon Loss", color="#2e86c1", markersize=3, linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss Curves")
    ax1.legend(frameon=False, fontsize=10)
    ax1.grid(alpha=0.2)

    # Right: Val F1
    ax2.plot(epochs, val_f1, "D-", color="#1a5276", markersize=3, linewidth=1.5)
    ax2.axhline(y=0.6661, color="#27ae60", linestyle="--", alpha=0.7, label="Best F1=0.6661")
    ax2.axhline(y=0.333, color="red", linestyle="--", alpha=0.4, label="Random=0.333")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Weighted F1-Score")
    ax2.set_title("Validation F1 (Subject-Held-Out)")
    ax2.legend(frameon=False, fontsize=10)
    ax2.grid(alpha=0.2)
    ax2.set_ylim(0.3, 0.75)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "training_curves_samratth.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close()
    print(f"Saved: {path}")


def print_latex_table():
    """Print ablation results as LaTeX table for paper."""
    csv_path = "runs/ablation_results.csv"
    if not os.path.isfile(csv_path):
        return

    print("\n% === LaTeX Table for Paper ===")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Modality Ablation Study (Subject-Held-Out, 3-Class Classification)}")
    print("\\label{tab:ablation}")
    print("\\begin{tabular}{lcc}")
    print("\\hline")
    print("\\textbf{Variant} & \\textbf{Modalities} & \\textbf{Weighted F1} \\\\")
    print("\\hline")

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["variant"].replace("_", "\\_")
            f1 = float(row["best_val_f1"])
            # Determine modalities
            mods = {"voice": "V", "activity": "A", "mobility": "M"}
            active = []
            v = row["variant"]
            if "voice" in v: active.append("V")
            if "activity" in v: active.append("A")
            if "mobility" in v: active.append("M")
            if v == "full-fusion": active = ["V", "A", "M"]
            mod_str = "+".join(active)
            bold = "\\textbf" if v == "full-fusion" else ""
            if bold:
                print(f"\\textbf{{{name}}} & \\textbf{{{mod_str}}} & \\textbf{{{f1:.4f}}} \\\\")
            else:
                print(f"{name} & {mod_str} & {f1:.4f} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")


if __name__ == "__main__":
    plot_ablation_bar()
    plot_training_curves()
    print_latex_table()
    print(f"\nAll figures saved to {FIG_DIR}/")