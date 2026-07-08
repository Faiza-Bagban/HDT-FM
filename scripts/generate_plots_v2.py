# generate_plots_v2.py — publication-grade redesign, no model loading needed
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyArrowPatch
import seaborn as sns

os.makedirs("results/figures", exist_ok=True)

# ── Global publication style ─────────────────────────────────────────────────
# plt.rcParams.update({
#     "font.family":        "sans-serif",
#     "font.sans-serif":    ["DejaVu Sans", "Arial", "Helvetica"],
#     "font.size":           13,
#     "axes.titlesize":      15,
#     "axes.titleweight":   "bold",
#     "axes.labelsize":      14,
#     "axes.labelweight":   "bold",
#     "xtick.labelsize":     13,
#     "ytick.labelsize":     13,
#     "xtick.color":        "#2C2C2A",
#     "ytick.color":        "#2C2C2A",
#     "axes.edgecolor":     "#2C2C2A",
#     "axes.linewidth":      1.1,
#     "legend.fontsize":     12,
#     "legend.frameon":      True,
#     "legend.framealpha":   0.95,
#     "legend.edgecolor":   "#B4B2A9",
#     "figure.facecolor":   "white",
#     "axes.facecolor":     "white",
#     "savefig.facecolor":  "white",
#     "axes.grid":           True,
#     "grid.alpha":          0.25,
#     "grid.linewidth":      0.6,
#     "grid.linestyle":     "-",
# })

plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size":           14,
    "font.weight":        "bold",
    "axes.titlesize":      14,
    "axes.titleweight":   "bold",
    "axes.labelsize":      14,
    "axes.labelweight":   "bold",
    "xtick.labelsize":     14,
    "ytick.labelsize":     14,
    "xtick.color":        "#2C2C2A",
    "ytick.color":        "#2C2C2A",
    "axes.edgecolor":     "#2C2C2A",
    "axes.linewidth":      1.1,
    "legend.fontsize":     14,
    })

plt.rcParams["font.weight"] = "bold"
for tick_group in ["xtick", "ytick"]:
    plt.rcParams[f"{tick_group}.labelsize"] = 14
# Colorblind-safe palette (Tidepool-style)
BLUE    = "#2a78d6"
AQUA    = "#1baf7a"
YELLOW  = "#eda100"
VIOLET  = "#4a3aa7"
RED     = "#e34948"
GRAY    = "#898781"
DARKINK = "#1a1a19"

def style_spines(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_linewidth(1.1)
        ax.spines[s].set_color("#2C2C2A")
    ax.tick_params(width=1.0, length=5)

def save(fig, name):
    fig.savefig(f"results/figures/{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"\u2713 {name}")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Confusion Matrix (clean sequential blue, no gridlines inside cells)
# ═════════════════════════════════════════════════════════════════════════════
cm = np.array([
    [10370, 3068,  952],
    [2160,  1491,  959],
    [781,   1247, 6568],
])
labels = ["Baseline", "Amusement", "Stress"]
cm_norm = cm / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(6.5, 5.8))
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="equal")

for i in range(3):
    for j in range(3):
        val = cm[i, j]
        pct = cm_norm[i, j] * 100
        textcolor = "white" if cm_norm[i, j] > 0.55 else "#1a1a19"
        ax.text(j, i, f"{val:,}\n{pct:.1f}%", ha="center", va="center",
                fontsize=14, fontweight="bold", color=textcolor)

ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=14, fontweight="bold")
ax.set_yticks(range(3)); ax.set_yticklabels(labels, fontsize=14, fontweight="bold")
ax.set_xlabel("Predicted label", labelpad=10)
ax.set_ylabel("True label", labelpad=10)
for s in ax.spines.values():
    s.set_visible(True); s.set_linewidth(1.2); s.set_color("#2C2C2A")
ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=2.5)
ax.grid(which="major", visible=False)
ax.tick_params(which="minor", size=0)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Row-normalized proportion", fontsize=14, fontweight="bold")
cbar.ax.tick_params(labelsize=14)

for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")

fig.tight_layout()
save(fig, "confusion_matrix")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Training loss + val F1 (dual axis, clean lines, marker on best epoch)
# ═════════════════════════════════════════════════════════════════════════════
losses  = [0.4649,0.2232,0.1556,0.1287,0.1146,0.1078,0.0994,0.0921,0.0960,0.0865,
           0.0848,0.0815,0.0760,0.0739,0.0757,0.0913,0.0704,0.0612,0.0540,0.0444,
           0.0404,0.0369,0.0310,0.0224,0.0219,0.0197,0.0185,0.0160,0.0149,0.0139]
val_f1s = [0.5990,0.5579,0.6300,0.5597,0.6489,0.6244,0.6511,0.6365,0.6624,0.6527,
           0.6623,0.6761,0.6635,0.6697,0.6364,0.6312,0.6385,0.6335,0.6168,0.6056,
           0.6137,0.6227,0.6329,0.6307,0.6340,0.6300,0.6371,0.6439,0.6389,0.6382]
epochs = list(range(30))
best_epoch = int(np.argmax(val_f1s))

fig, ax1 = plt.subplots(figsize=(8.5, 5))
ax2 = ax1.twinx()

ax1.plot(epochs, losses, color=RED, linewidth=2.2, label="Training loss", zorder=3)
ax2.plot(epochs, val_f1s, color=BLUE, linewidth=2.2, linestyle="--",
         label="Validation F1 (subject-held-out)", zorder=3)
ax2.scatter([best_epoch], [val_f1s[best_epoch]], s=90, color=BLUE,
            edgecolor="white", linewidth=1.5, zorder=5)
# ax2.annotate(f"Best: F1={val_f1s[best_epoch]:.4f}\n(epoch {best_epoch})",
#              xy=(best_epoch, val_f1s[best_epoch]),
#              xytext=(best_epoch + 4, val_f1s[best_epoch] + 0.025),
#              fontsize=14, fontweight="bold", color=BLUE,
#              arrowprops=dict(arrowstyle="-", color=BLUE, lw=1))

# trial
ax2.annotate(f"Best: F1={val_f1s[best_epoch]:.4f}\n(epoch {best_epoch})",
             xy=(best_epoch, val_f1s[best_epoch]),
             xytext=(best_epoch + 3, val_f1s[best_epoch] + 0.03),
             fontsize=12, fontweight="bold", color=BLUE,
             arrowprops=dict(arrowstyle="-", color=BLUE, lw=1))

ax1.set_xlabel("Epoch")
ax1.set_ylabel("Training loss", color=RED)
ax2.set_ylabel("Validation F1", color=BLUE)
ax1.tick_params(axis="y", labelcolor=RED, labelsize=14)
ax2.tick_params(axis="y", labelcolor=BLUE, labelsize=14)
for lbl in ax1.get_xticklabels() + ax1.get_yticklabels() + ax2.get_yticklabels():
    lbl.set_fontweight("bold")
style_spines(ax1)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_linewidth(1.1)

# lines1, labs1 = ax1.get_legend_handles_labels()
# lines2, labs2 = ax2.get_legend_handles_labels()
# ax1.legend(lines1 + lines2, labs1 + labs2, loc="center right", fontsize=11)

# lines1, labs1 = ax1.get_legend_handles_labels()
# lines2, labs2 = ax2.get_legend_handles_labels()
# ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper right",
#            bbox_to_anchor=(0.98, 0.62), fontsize=11)

lines1, labs1 = ax1.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
# ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper right",
#            bbox_to_anchor=(0.99, 0.99), bbox_transform=ax1.transAxes,
#            fontsize=14, framealpha=0.95)
# leg = ax1.legend(lines1 + lines2, labs1 + labs2, loc="lower right",
#            bbox_to_anchor=(0.99, 0.32), bbox_transform=ax1.transAxes,
#            fontsize=12, framealpha=0.95)

# leg = ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper center",
#            bbox_to_anchor=(0.5, -0.15), bbox_transform=ax1.transAxes,
#            fontsize=12, framealpha=0.95, ncol=2)

leg = ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper center",
           bbox_to_anchor=(0.5, -0.28), bbox_transform=ax1.transAxes,
           fontsize=12, framealpha=0.95, ncol=2)
for t in leg.get_texts():
    t.set_fontweight("bold")
for t in leg.get_texts():
    t.set_fontweight("bold")
for t in leg.get_texts():
    t.set_fontweight("bold")

# fig.tight_layout()
fig.subplots_adjust(bottom=0.32)
save(fig, "loss_curve")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Ablation bar (horizontal, sequential blue, direct value labels)
# ═════════════════════════════════════════════════════════════════════════════
variants = ["Voice only", "Activity only\n(HRV)", "Voice + Activity", "Full fusion\n(V+A+M, proposed)"]
f1_vals  = [0.1480, 0.6679, 0.6779, 0.6779]
colors_ab = ["#B4B2A9", "#85B7EB", "#378ADD", BLUE]

fig, ax = plt.subplots(figsize=(8, 5))
y_pos = np.arange(len(variants))
bars = ax.barh(y_pos, f1_vals, color=colors_ab, edgecolor="#1a1a19", linewidth=0.8, height=0.6)

for bar, val in zip(bars, f1_vals):
    ax.text(val + 0.015, bar.get_y() + bar.get_height()/2, f"{val:.4f}",
            va="center", fontsize=14, fontweight="bold", color="#1a1a19")

# ax.axvline(x=0.167, color=GRAY, linestyle=":", linewidth=1.5, zorder=1)
# ax.text(0.167, len(variants) - 0.3, "  Random\n  baseline", fontsize=10,
#         color=GRAY, va="top", fontweight="bold")

ax.axvline(x=0.167, color=GRAY, linestyle=":", linewidth=1.5, zorder=1)
ax.text(0.167, -0.75, "Random\nbaseline", fontsize=12,
        color=GRAY, va="top", ha="center", fontweight="bold", clip_on=False)

ax.set_yticks(y_pos); ax.set_yticklabels(variants, fontsize=14, fontweight="bold")
for lbl in ax.get_xticklabels():
    lbl.set_fontweight("bold")
ax.set_xlabel("Weighted F1 score")
# ax.set_xlim(0, 0.85)
ax.set_xlim(0, 0.85)
ax.margins(y=0.15)
fig.subplots_adjust(bottom=0.24)
ax.invert_yaxis()
style_spines(ax)
ax.grid(axis="y", visible=False)

fig.tight_layout()
save(fig, "ablation_bar")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Per-class precision/recall/F1 (grouped bar, categorical palette)
# ═════════════════════════════════════════════════════════════════════════════
classes   = ["Baseline", "Amusement", "Stress"]
precision = [0.78, 0.26, 0.77]
recall_v  = [0.72, 0.32, 0.76]
f1_per    = [0.75, 0.29, 0.77]

x = np.arange(len(classes))
width = 0.25
fig, ax = plt.subplots(figsize=(8, 5.2))
ax.bar(x - width, precision, width, label="Precision", color=BLUE,   edgecolor="#1a1a19", linewidth=0.7)
ax.bar(x,         recall_v,  width, label="Recall",    color=AQUA,   edgecolor="#1a1a19", linewidth=0.7)
ax.bar(x + width, f1_per,    width, label="F1-score",  color=VIOLET, edgecolor="#1a1a19", linewidth=0.7)

for i, (p, r, f) in enumerate(zip(precision, recall_v, f1_per)):
    ax.text(i - width, p + 0.02, f"{p:.2f}", ha="center", fontsize=14, fontweight="bold")
    ax.text(i,         r + 0.02, f"{r:.2f}", ha="center", fontsize=14, fontweight="bold")
    ax.text(i + width, f + 0.02, f"{f:.2f}", ha="center", fontsize=14, fontweight="bold")

ax.set_xticks(x); 
ax.set_xticklabels(classes, fontsize=14, fontweight="bold")
for lbl in ax.get_yticklabels():
    lbl.set_fontweight("bold")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.0)
# ax.legend(loc="upper right", ncol=1)

# ax.legend(loc="upper right", ncol=1, bbox_to_anchor=(1.0, 1.15))
leg = ax.legend(loc="upper right", ncol=1, bbox_to_anchor=(1.0, 1.15), fontsize=14)
for t in leg.get_texts():
    t.set_fontweight("bold")
# ax.set_ylim(0, 1.12)
# ax.set_ylim(0, 1.22)
ax.set_ylim(0, 1.05)
style_spines(ax)

fig.tight_layout()
save(fig, "per_class_f1")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Class distribution (train vs val, small multiples)
# ═════════════════════════════════════════════════════════════════════════════
label_names  = ["Baseline", "Amusement", "Stress"]
train_counts = [57250, 18454, 32350]
val_counts   = [14390, 4610,  8596]
colors_dist  = [BLUE, YELLOW, RED]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=False)

for ax, counts, title, n in zip(
    axes, [train_counts, val_counts],
    ["Training set  (S2\u2013S14, 12 subjects)", "Validation set  (S15\u2013S17, held-out)"],
    [sum(train_counts), sum(val_counts)]
):
    bars = ax.bar(label_names, counts, color=colors_dist, edgecolor="#1a1a19", linewidth=0.8, width=0.6)
    for bar, v in zip(bars, counts):
        pct = 100 * v / n
        # ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.02,
        #         f"{v:,}\n({pct:.0f}%)", ha="center", fontsize=11, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.02,
        f"{v:,}\n({pct:.0f}%)", ha="center", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Sample count")
    style_spines(ax)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    ax.set_ylim(0, max(counts) * 1.22)

fig.tight_layout()
save(fig, "class_distribution")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Subject-wise F1 (emphasis coloring: train=gray, held-out=blue)
# ═════════════════════════════════════════════════════════════════════════════
all_subjects = [f"S{i}" for i in [2,3,4,5,6,7,8,9,10,11,13,14,15,16,17]]
f1_by_subj   = [0.99,0.99,0.98,0.99,0.99,0.98,0.99,0.99,0.98,0.99,0.99,0.98,
                0.5069, 0.8630, 0.6532]
is_heldout   = [False]*12 + [True]*3
colors_subj  = [BLUE if h else "#C3C2B7" for h in is_heldout]

fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(all_subjects, f1_by_subj, color=colors_subj, edgecolor="#1a1a19", linewidth=0.7, width=0.65)

mean_heldout = np.mean([f for f, h in zip(f1_by_subj, is_heldout) if h])
ax.axhline(y=mean_heldout, color=RED, linestyle="--", linewidth=1.6, zorder=1)
# ax.text(14.55, mean_heldout + 0.02, f"Mean held-out F1 = {mean_heldout:.4f}",
#         fontsize=11, fontweight="bold", color=RED, ha="right")

# ax.text(6.5, mean_heldout + 0.03, f"Mean held-out F1 = {mean_heldout:.4f}",
#         fontsize=11, fontweight="bold", color=RED, ha="center")

ax.text(5, mean_heldout - 0.05, f"Mean held-out F1 = {mean_heldout:.4f}",
        fontsize=14, fontweight="bold", color=RED, ha="center", va="top")

for bar, val in zip(bars, f1_by_subj):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
            f"{val:.2f}", ha="center", fontsize=14, fontweight="bold")

ax.set_xlabel("Subject ID")
ax.set_ylabel("Weighted F1")
ax.set_ylim(0, 1.12)
style_spines(ax)
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontweight("bold")
ax.grid(axis="x", visible=False)

from matplotlib.patches import Patch
legend_elems = [
    Patch(facecolor="#C3C2B7", edgecolor="#1a1a19", label="Training subjects"),
    Patch(facecolor=BLUE, edgecolor="#1a1a19", label="Held-out subjects (S15\u2013S17)"),
]
# ax.legend(handles=legend_elems, loc="lower left", fontsize=11)
# ax.legend(handles=legend_elems, loc="upper right", fontsize=11)
# ax.legend(handles=legend_elems, loc="upper right",
#           bbox_to_anchor=(1.0, 1.0), fontsize=11, framealpha=0.95)

ax.legend(handles=legend_elems, loc="lower left",
          bbox_to_anchor=(0, 1.02, 1, 0.1), ncol=2,
          mode="expand", borderaxespad=0, fontsize=14)
for t in leg.get_texts():
    t.set_fontweight("bold")

fig.tight_layout()
save(fig, "subject_f1")

print("\n\u2705 All 6 publication-quality figures saved to results/figures/ at 600 DPI")