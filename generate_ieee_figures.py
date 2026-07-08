# generate_ieee_figures.py
"""
Publication-Quality IEEE Figures for HDT-FM Paper.
All figures: 600 DPI, font 14 bold, colorblind-friendly, vector-quality.
Generates .png files in results/figures/

Figures:
  1. Confusion Matrix (annotated heatmap)
  2. Training Dynamics (dual-panel: loss curves + val F1)
  3. Ablation Study (horizontal lollipop chart)
  4. Per-Class Performance (grouped bar + radar overlay)
  5. Subject-Wise Generalization (bar + variance band)
  6. Class Distribution (waffle-style proportional chart)
  7. ROC Curves (per-class + micro-average)
  8. Twin Reconstruction Loss Convergence
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

# ══════════════════════════════════════════════════════════════
# GLOBAL STYLE — IEEE / NeurIPS / Nature grade
# ══════════════════════════════════════════════════════════════

# Colorblind-friendly palette (Wong 2011 + custom)
C_BLUE    = "#0077BB"
C_ORANGE  = "#EE7733"
C_GREEN   = "#009988"
C_RED     = "#CC3311"
C_PURPLE  = "#AA3377"
C_CYAN    = "#33BBEE"
C_GREY    = "#BBBBBB"
C_DARK    = "#1A1A2E"
C_BG      = "#FAFAFA"

CLASS_COLORS = [C_BLUE, C_ORANGE, C_GREEN]
CLASS_NAMES  = ["Baseline", "Amusement", "Stress"]

plt.rcParams.update({
    "font.size": 14,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.5,
})

FIG_DIR = "results/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# DATA (from actual experimental runs)
# ══════════════════════════════════════════════════════════════

# Confusion matrix (full-fusion, real voice, subject-held-out)
CM = np.array([
    [10370, 3068,  952],
    [ 2160, 1491,  959],
    [  781, 1247, 6568],
])

# Training losses (30 epochs, v3 real-voice run)
TRAIN_LOSS = [
    0.4649, 0.2232, 0.1556, 0.1287, 0.1146, 0.1078, 0.0994,
    0.0921, 0.0960, 0.0865, 0.0848, 0.0815, 0.0760, 0.0739,
    0.0757, 0.0913, 0.0704, 0.0612, 0.0540, 0.0444, 0.0404,
    0.0369, 0.0310, 0.0224, 0.0219, 0.0197, 0.0185, 0.0160,
    0.0149, 0.0139,
]

CE_LOSS = [
    0.4540, 0.2152, 0.1486, 0.1223, 0.1085, 0.1015, 0.0934,
    0.0863, 0.0901, 0.0810, 0.0794, 0.0765, 0.0712, 0.0692,
    0.0710, 0.0866, 0.0660, 0.0570, 0.0504, 0.0410, 0.0373,
    0.0339, 0.0282, 0.0197, 0.0194, 0.0172, 0.0161, 0.0137,
    0.0125, 0.0116,
]

TWIN_LOSS = [
    0.1095, 0.0794, 0.0700, 0.0635, 0.0614, 0.0632, 0.0599,
    0.0587, 0.0589, 0.0549, 0.0545, 0.0500, 0.0476, 0.0466,
    0.0470, 0.0477, 0.0433, 0.0415, 0.0364, 0.0345, 0.0314,
    0.0295, 0.0278, 0.0263, 0.0255, 0.0249, 0.0242, 0.0238,
    0.0235, 0.0235,
]

VAL_F1 = [
    0.5990, 0.5579, 0.6300, 0.5597, 0.6489, 0.6244, 0.6511,
    0.6365, 0.6624, 0.6527, 0.6623, 0.6761, 0.6635, 0.6697,
    0.6364, 0.6312, 0.6385, 0.6335, 0.6168, 0.6056, 0.6137,
    0.6227, 0.6329, 0.6307, 0.6340, 0.6300, 0.6371, 0.6439,
    0.6389, 0.6382,
]

# Ablation results (v2 with real voice)
ABLATION = {
    "Voice Only":        0.1480,
    "Mobility Only":     0.1480,
    "Activity Only":     0.6679,
    "Voice+Mobility":    0.1480,
    "Activity+Mobility": 0.6679,
    "Voice+Activity":    0.6779,
    "Full Fusion":       0.6779,
}

# Per-class metrics (full-fusion)
PER_CLASS = {
    "Baseline":  {"precision": 0.78, "recall": 0.72, "f1": 0.75, "support": 14390},
    "Amusement": {"precision": 0.26, "recall": 0.32, "f1": 0.29, "support": 4610},
    "Stress":    {"precision": 0.77, "recall": 0.76, "f1": 0.77, "support": 8596},
}

# Subject-wise F1
SUBJECT_F1 = {15: 0.5069, 16: 0.8630, 17: 0.6532}

# AUC-ROC
AUC_ROC = 0.8112


# ══════════════════════════════════════════════════════════════
# FIGURE 1: CONFUSION MATRIX
# ══════════════════════════════════════════════════════════════

def fig_confusion_matrix():
    fig, ax = plt.subplots(figsize=(7, 6))

    # Normalize for color mapping
    cm_norm = CM.astype(float) / CM.sum(axis=1, keepdims=True)

    # Custom colormap: white → deep blue
    cmap = LinearSegmentedColormap.from_list("ieee", ["#FFFFFF", "#B3D9FF", "#4DA6FF", "#0066CC", "#003366"])

    im = ax.imshow(cm_norm, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    # Annotate each cell with count + percentage
    for i in range(3):
        for j in range(3):
            count = CM[i, j]
            pct = cm_norm[i, j] * 100
            color = "white" if cm_norm[i, j] > 0.5 else C_DARK
            ax.text(j, i, f"{count:,}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=13, fontweight="bold",
                    color=color,
                    path_effects=[pe.withStroke(linewidth=0.5, foreground="white")])

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(CLASS_NAMES, fontweight="bold")
    ax.set_yticklabels(CLASS_NAMES, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=14, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=14, fontweight="bold")
    ax.set_title("Confusion Matrix — Full Fusion (Subject-Held-Out)", fontsize=15, fontweight="bold", pad=12)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proportion", fontsize=12, fontweight="bold")
    cbar.ax.tick_params(labelsize=11)

    # Border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)

    save(fig, "fig1_confusion_matrix")


# ══════════════════════════════════════════════════════════════
# FIGURE 2: TRAINING DYNAMICS (DUAL PANEL)
# ══════════════════════════════════════════════════════════════

def fig_training_dynamics():
    epochs = np.arange(30)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── Left: Loss Curves ──
    ax1.plot(epochs, CE_LOSS, "-o", color=C_RED, markersize=4, linewidth=2,
             label="Cross-Entropy Loss", zorder=3)
    ax1.plot(epochs, TWIN_LOSS, "-s", color=C_BLUE, markersize=4, linewidth=2,
             label="Twin Reconstruction Loss", zorder=3)
    ax1.fill_between(epochs, CE_LOSS, alpha=0.08, color=C_RED)
    ax1.fill_between(epochs, TWIN_LOSS, alpha=0.08, color=C_BLUE)

    ax1.set_xlabel("Epoch", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Loss", fontsize=14, fontweight="bold")
    ax1.set_title("(a) Training Loss Convergence", fontsize=15, fontweight="bold")
    ax1.legend(frameon=True, fancybox=True, shadow=False, edgecolor=C_GREY,
               fontsize=11, loc="upper right")
    ax1.set_xlim(-0.5, 29.5)
    ax1.set_ylim(0, 0.50)

    # Annotate final values
    ax1.annotate(f"CE={CE_LOSS[-1]:.4f}", xy=(29, CE_LOSS[-1]),
                 xytext=(22, 0.12), fontsize=10, fontweight="bold", color=C_RED,
                 arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.2))
    ax1.annotate(f"Twin={TWIN_LOSS[-1]:.4f}", xy=(29, TWIN_LOSS[-1]),
                 xytext=(22, 0.08), fontsize=10, fontweight="bold", color=C_BLUE,
                 arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1.2))

    # ── Right: Val F1 ──
    ax2.plot(epochs, VAL_F1, "-D", color=C_GREEN, markersize=4, linewidth=2,
             label="Weighted F1-Score", zorder=3)
    ax2.fill_between(epochs, VAL_F1, alpha=0.08, color=C_GREEN)

    best_epoch = np.argmax(VAL_F1)
    best_f1 = max(VAL_F1)
    ax2.axhline(y=best_f1, color=C_PURPLE, linestyle="--", linewidth=1.2, alpha=0.7,
                label=f"Best F1 = {best_f1:.4f} (Epoch {best_epoch})")
    ax2.axhline(y=0.333, color=C_RED, linestyle=":", linewidth=1, alpha=0.5,
                label="Random Baseline (0.333)")
    ax2.plot(best_epoch, best_f1, "*", color=C_PURPLE, markersize=16, zorder=5)

    ax2.set_xlabel("Epoch", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Weighted F1-Score", fontsize=14, fontweight="bold")
    ax2.set_title("(b) Validation Performance", fontsize=15, fontweight="bold")
    ax2.legend(frameon=True, fancybox=True, shadow=False, edgecolor=C_GREY,
               fontsize=10, loc="lower right")
    ax2.set_xlim(-0.5, 29.5)
    ax2.set_ylim(0.30, 0.75)

    fig.tight_layout(w_pad=3)
    save(fig, "fig2_training_dynamics")


# ══════════════════════════════════════════════════════════════
# FIGURE 3: ABLATION STUDY (LOLLIPOP CHART)
# ══════════════════════════════════════════════════════════════

def fig_ablation():
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Sort by F1
    sorted_items = sorted(ABLATION.items(), key=lambda x: x[1])
    names = [k for k, v in sorted_items]
    f1s = [v for k, v in sorted_items]

    y_pos = np.arange(len(names))

    # Color: gradient based on performance
    colors = []
    for f1 in f1s:
        if f1 < 0.2:
            colors.append(C_GREY)
        elif f1 < 0.5:
            colors.append(C_ORANGE)
        elif f1 < 0.65:
            colors.append(C_CYAN)
        else:
            colors.append(C_BLUE)
    # Full fusion = highlight
    if names[-1] == "Full Fusion":
        colors[-1] = C_GREEN

    # Lollipop stems
    for i, (y, f1, c) in enumerate(zip(y_pos, f1s, colors)):
        ax.hlines(y, 0, f1, color=c, linewidth=2.5, zorder=2)
        ax.plot(f1, y, "o", color=c, markersize=12, zorder=3,
                markeredgecolor="white", markeredgewidth=1.5)

    # Value labels
    for y, f1 in zip(y_pos, f1s):
        ax.text(f1 + 0.015, y, f"{f1:.4f}", va="center", fontsize=12,
                fontweight="bold", color=C_DARK)

    # Random baseline
    ax.axvline(x=0.333, color=C_RED, linestyle="--", linewidth=1.2, alpha=0.6,
               label="Random Baseline (0.333)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontweight="bold", fontsize=12)
    ax.set_xlabel("Weighted F1-Score", fontsize=14, fontweight="bold")
    ax.set_title("Modality Ablation Study", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlim(0, 0.82)
    ax.legend(fontsize=11, loc="lower right", frameon=True, edgecolor=C_GREY)

    # Clean up
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    save(fig, "fig3_ablation_lollipop")


# ══════════════════════════════════════════════════════════════
# FIGURE 4: PER-CLASS PERFORMANCE (GROUPED BAR)
# ══════════════════════════════════════════════════════════════

def fig_per_class():
    fig, ax = plt.subplots(figsize=(9, 5.5))

    classes = list(PER_CLASS.keys())
    metrics = ["precision", "recall", "f1"]
    metric_labels = ["Precision", "Recall", "F1-Score"]
    metric_colors = [C_BLUE, C_ORANGE, C_GREEN]

    x = np.arange(len(classes))
    width = 0.22

    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, metric_colors)):
        vals = [PER_CLASS[c][metric] for c in classes]
        bars = ax.bar(x + i * width, vals, width, label=label, color=color,
                      edgecolor="white", linewidth=1, zorder=3)
        # Value on top
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=11,
                    fontweight="bold", color=color)

    # Support annotation
    for i, c in enumerate(classes):
        support = PER_CLASS[c]["support"]
        ax.text(x[i] + width, -0.08, f"n={support:,}", ha="center",
                fontsize=10, color=C_GREY, fontstyle="italic")

    ax.set_xticks(x + width)
    ax.set_xticklabels(classes, fontweight="bold", fontsize=13)
    ax.set_ylabel("Score", fontsize=14, fontweight="bold")
    ax.set_xlabel("Class", fontsize=14, fontweight="bold")
    ax.set_title("Per-Class Performance — Full Fusion", fontsize=15, fontweight="bold", pad=12)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=True, edgecolor=C_GREY, fontsize=12, loc="upper right")

    save(fig, "fig4_per_class_performance")


# ══════════════════════════════════════════════════════════════
# FIGURE 5: SUBJECT-WISE GENERALIZATION
# ══════════════════════════════════════════════════════════════

def fig_subject_wise():
    fig, ax = plt.subplots(figsize=(8, 5.5))

    subjects = sorted(SUBJECT_F1.keys())
    f1s = [SUBJECT_F1[s] for s in subjects]
    labels = [f"S{s}" for s in subjects]

    x = np.arange(len(subjects))

    # Color by performance
    colors = []
    for f1 in f1s:
        if f1 < 0.55:
            colors.append(C_RED)
        elif f1 < 0.70:
            colors.append(C_ORANGE)
        else:
            colors.append(C_GREEN)

    bars = ax.bar(x, f1s, color=colors, edgecolor="white", linewidth=1.5,
                  width=0.5, zorder=3)

    # Value labels
    for bar, f1 in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{f1:.4f}", ha="center", fontsize=13, fontweight="bold", color=C_DARK)

    # Mean line
    mean_f1 = np.mean(f1s)
    ax.axhline(y=mean_f1, color=C_PURPLE, linestyle="--", linewidth=1.5,
               label=f"Mean F1 = {mean_f1:.4f}", zorder=2)

    # Variance band
    std_f1 = np.std(f1s)
    ax.axhspan(mean_f1 - std_f1, mean_f1 + std_f1, alpha=0.08, color=C_PURPLE,
               label=f"$\\pm$1 SD ({std_f1:.4f})")

    # AUC annotation
    ax.text(0.98, 0.05, f"Weighted AUC-ROC = {AUC_ROC:.4f}",
            transform=ax.transAxes, ha="right", fontsize=12,
            fontweight="bold", color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F4FD", edgecolor=C_BLUE, alpha=0.9))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold", fontsize=14)
    ax.set_ylabel("Weighted F1-Score", fontsize=14, fontweight="bold")
    ax.set_xlabel("Held-Out Subject", fontsize=14, fontweight="bold")
    ax.set_title("Subject-Wise Generalization (Held-Out Evaluation)",
                 fontsize=15, fontweight="bold", pad=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11, frameon=True, edgecolor=C_GREY, loc="upper left")

    save(fig, "fig5_subject_wise_f1")


# ══════════════════════════════════════════════════════════════
# FIGURE 6: CLASS DISTRIBUTION
# ══════════════════════════════════════════════════════════════

def fig_class_distribution():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.2, 1]})

    supports = [14390, 4610, 8596]
    total = sum(supports)
    pcts = [s / total * 100 for s in supports]

    # ── Left: Horizontal bar ──
    y = np.arange(3)
    bars = ax1.barh(y, supports, color=CLASS_COLORS, edgecolor="white",
                    height=0.5, linewidth=1.5, zorder=3)

    for bar, s, p in zip(bars, supports, pcts):
        ax1.text(bar.get_width() + 200, bar.get_y() + bar.get_height() / 2,
                 f"{s:,}  ({p:.1f}%)", va="center", fontsize=12, fontweight="bold")

    ax1.set_yticks(y)
    ax1.set_yticklabels(CLASS_NAMES, fontweight="bold", fontsize=13)
    ax1.set_xlabel("Number of Samples", fontsize=14, fontweight="bold")
    ax1.set_title("(a) Validation Set Distribution", fontsize=15, fontweight="bold")
    ax1.set_xlim(0, max(supports) * 1.35)
    ax1.spines["left"].set_visible(False)
    ax1.tick_params(axis="y", length=0)

    # Imbalance ratio annotation
    ratio = max(supports) / min(supports)
    ax1.text(0.5, -0.12, f"Imbalance Ratio: {ratio:.1f}:1 (max/min)",
             transform=ax1.transAxes, ha="center", fontsize=11,
             fontstyle="italic", color=C_DARK)

    # ── Right: Donut chart ──
    wedges, texts, autotexts = ax2.pie(
        supports, labels=CLASS_NAMES, colors=CLASS_COLORS,
        autopct="%1.1f%%", startangle=90, pctdistance=0.78,
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
        textprops=dict(fontsize=12, fontweight="bold"),
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")

    ax2.set_title("(b) Class Proportions", fontsize=15, fontweight="bold")

    # Center text
    ax2.text(0, 0, f"N={total:,}", ha="center", va="center",
             fontsize=14, fontweight="bold", color=C_DARK)

    fig.tight_layout(w_pad=3)
    save(fig, "fig6_class_distribution")


# ══════════════════════════════════════════════════════════════
# FIGURE 7: MODALITY CONTRIBUTION HEATMAP
# ══════════════════════════════════════════════════════════════

def fig_modality_heatmap():
    fig, ax = plt.subplots(figsize=(7, 5))

    # Build modality presence matrix
    modalities = ["Voice", "Activity", "Mobility"]
    variants = list(ABLATION.keys())
    f1_vals = list(ABLATION.values())

    # Modality presence: 1 if active, 0 if not
    presence = {
        "Voice Only":        [1, 0, 0],
        "Activity Only":     [0, 1, 0],
        "Mobility Only":     [0, 0, 1],
        "Voice+Mobility":    [1, 0, 1],
        "Voice+Activity":    [1, 1, 0],
        "Activity+Mobility": [0, 1, 1],
        "Full Fusion":       [1, 1, 1],
    }

    # Sort by F1
    sorted_v = sorted(ABLATION.items(), key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    y_labels = []
    for idx, (variant, f1) in enumerate(sorted_v):
        y = len(sorted_v) - 1 - idx
        y_labels.append(f"{variant} (F1={f1:.4f})")
        pres = presence[variant]
        for j, (mod, active) in enumerate(zip(modalities, pres)):
            if active:
                circle = plt.Circle((j, y), 0.3, color=CLASS_COLORS[j],
                                    alpha=0.85, zorder=3)
                ax.add_patch(circle)
                ax.text(j, y, mod[0], ha="center", va="center",
                        fontsize=11, fontweight="bold", color="white", zorder=4)
            else:
                circle = plt.Circle((j, y), 0.3, color="#EEEEEE",
                                    linewidth=1.5, linestyle="--",
                                    fill=True, zorder=2)
                ax.add_patch(circle)

    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(-0.6, len(sorted_v) - 0.4)
    ax.set_xticks(range(3))
    ax.set_xticklabels(modalities, fontweight="bold", fontsize=13)
    ax.set_yticks(range(len(sorted_v)))
    ax.set_yticklabels(list(reversed(y_labels)), fontweight="bold", fontsize=11)
    ax.set_xlabel("Modality", fontsize=14, fontweight="bold")
    ax.set_title("Modality Contribution Map", fontsize=15, fontweight="bold", pad=12)
    ax.set_aspect("equal")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    save(fig, "fig7_modality_heatmap")


# ══════════════════════════════════════════════════════════════
# FIGURE 8: TWIN RECONSTRUCTION CONVERGENCE
# ══════════════════════════════════════════════════════════════

def fig_twin_convergence():
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = np.arange(30)

    ax.plot(epochs, TWIN_LOSS, "-o", color=C_BLUE, markersize=5, linewidth=2.5,
            label="Twin Reconstruction Loss (Huber)", zorder=3)
    ax.fill_between(epochs, TWIN_LOSS, alpha=0.1, color=C_BLUE)

    # Convergence zone
    converge_start = 22  # roughly where it plateaus
    ax.axvspan(converge_start, 29, alpha=0.06, color=C_GREEN,
               label="Convergence Zone")
    ax.axhline(y=TWIN_LOSS[-1], color=C_GREEN, linestyle="--", linewidth=1,
               alpha=0.7, label=f"Final = {TWIN_LOSS[-1]:.4f}")

    # Reduction annotation
    reduction = (1 - TWIN_LOSS[-1] / TWIN_LOSS[0]) * 100
    ax.annotate(f"{reduction:.0f}% reduction",
                xy=(15, (TWIN_LOSS[0] + TWIN_LOSS[-1]) / 2),
                fontsize=13, fontweight="bold", color=C_DARK, ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3E0",
                          edgecolor=C_ORANGE, alpha=0.9))

    ax.set_xlabel("Epoch", fontsize=14, fontweight="bold")
    ax.set_ylabel("Huber Loss", fontsize=14, fontweight="bold")
    ax.set_title("Digital Twin Reconstruction Loss Convergence",
                 fontsize=15, fontweight="bold", pad=12)
    ax.legend(fontsize=11, frameon=True, edgecolor=C_GREY, loc="upper right")
    ax.set_xlim(-0.5, 29.5)
    ax.set_ylim(0, 0.13)

    save(fig, "fig8_twin_convergence")


# ══════════════════════════════════════════════════════════════
# UTILITY
# ══════════════════════════════════════════════════════════════

def save(fig, name):
    png_path = os.path.join(FIG_DIR, f"{name}.png")
    pdf_path = os.path.join(FIG_DIR, f"{name}.pdf")
    fig.savefig(png_path, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {png_path}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  HDT-FM Paper Figures — IEEE Publication Quality")
    print("  600 DPI | Font 14 Bold | Colorblind-Friendly")
    print("=" * 60)

    fig_confusion_matrix()
    fig_training_dynamics()
    fig_ablation()
    fig_per_class()
    fig_subject_wise()
    fig_class_distribution()
    fig_modality_heatmap()
    fig_twin_convergence()

    print(f"\n  All figures saved to {FIG_DIR}/")
    print("  Formats: .png + .pdf (vector)")
    print("=" * 60)