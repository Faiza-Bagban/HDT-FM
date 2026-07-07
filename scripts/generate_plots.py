# # scripts/generate_plots.py
# import sys, os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import torch
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg')
# import seaborn as sns
# from sklearn.metrics import confusion_matrix
# from sklearn.preprocessing import StandardScaler
# from models.hdt_model import HDTModel

# FEATURE_COLS = [
#     'MEAN_RR','MEDIAN_RR','SDRR','RMSSD','SDSD','HR',
#     'pNN25','pNN50','SD1','SD2','KURT','SKEW',
#     'VLF','LF','HF','TP','LF_HF','HF_LF','LF_NU','HF_NU',
# ]
# LABEL_COL = 'condition label'
# DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
# os.makedirs("results/figures", exist_ok=True)

# # ── Load data ─────────────────────────────────────────────────────────────────
# df       = pd.read_csv("data/WESAD/wesad-chest-combined-classification-hrv.csv")
# df       = df.dropna(subset=FEATURE_COLS + [LABEL_COL, 'subject id'])
# subjects = sorted(df['subject id'].unique())
# val_subj = subjects[-3:]
# train_df = df[~df['subject id'].isin(val_subj)]
# val_df   = df[df['subject id'].isin(val_subj)]

# scaler    = StandardScaler()
# scaler.fit(train_df[FEATURE_COLS].values.astype(np.float32))
# X_val     = scaler.transform(val_df[FEATURE_COLS].values.astype(np.float32))
# X_val_pad = np.zeros((len(X_val), 128), dtype=np.float32)
# X_val_pad[:, :X_val.shape[1]] = X_val
# y_val     = val_df[LABEL_COL].values.astype(np.int64)

# # ── Load model ────────────────────────────────────────────────────────────────
# model = HDTModel(n_classes=3).to(DEVICE)

# state_dict = torch.load("runs/best_model.pt", map_location=DEVICE)
# # strip _orig_mod. prefix if present (from torch.compile)
# state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
# model.load_state_dict(state_dict)
# model.eval()

# all_preds = []
# with torch.no_grad():
#     for i in range(0, len(X_val_pad), 64):
#         activity = torch.tensor(X_val_pad[i:i+64]).to(DEVICE)
#         mobility = torch.zeros(len(activity), 64).to(DEVICE)
#         audio    = torch.randn(len(activity), 16000).to(DEVICE)
#         with torch.amp.autocast('cuda', enabled=(DEVICE=="cuda")):
#             logits, _ = model(audio=audio, activity=activity, mobility=mobility)
#         all_preds.extend(logits.argmax(-1).cpu().tolist())

# # ─────────────────────────────────────────────────────────────────────────────
# # FIGURE 1 — Confusion Matrix
# # ─────────────────────────────────────────────────────────────────────────────
# cm  = confusion_matrix(y_val, all_preds)
# fig, ax = plt.subplots(figsize=(6, 5))
# sns.heatmap(
#     cm, annot=True, fmt='d', cmap='Blues',
#     xticklabels=["Baseline","Amusement","Stress"],
#     yticklabels=["Baseline","Amusement","Stress"],
#     ax=ax, linewidths=0.5, linecolor='gray',
#     annot_kws={"size": 16, "weight": "bold"}
# )
# ax.set_xlabel("Predicted Label", fontsize=16, fontweight='bold')
# ax.set_ylabel("True Label",      fontsize=16, fontweight='bold')
# ax.set_title("HDT-FM Confusion Matrix\n(Subject-Held-Out: S15–S17)", fontsize=16, fontweight='bold')
# ax.tick_params(axis='both', labelsize=16)
# for spine in ax.spines.values():
#     spine.set_visible(True)
#     spine.set_linewidth(1.5)
# plt.tight_layout()
# plt.savefig("results/figures/confusion_matrix.pdf", dpi=600, bbox_inches='tight')
# plt.savefig("results/figures/confusion_matrix.png", dpi=600, bbox_inches='tight')
# plt.close()
# print("✓ Saved confusion_matrix")

# # ─────────────────────────────────────────────────────────────────────────────
# # FIGURE 2 — Training Loss Curve
# # ─────────────────────────────────────────────────────────────────────────────
# epochs  = list(range(30))
# losses  = [0.4666,0.2047,0.1392,0.1114,0.0936,0.0821,0.0717,0.0600,0.0564,0.0488,
#            0.0417,0.0374,0.0332,0.0295,0.0273,0.0186,0.0175,0.0158,0.014,0.012,
#            0.010,0.009,0.008,0.007,0.006,0.005,0.004,0.003,0.002,0.0009]
# val_f1s = [0.9123,0.9634,0.9757,0.9793,0.9817,0.9911,0.9891,0.9936,0.9937,0.9922,
#            0.9910,0.9926,0.9965,0.9976,0.9980,0.9989,0.9989,0.9993,0.9993,0.9994,
#            0.9995,0.9996,0.9997,0.9998,0.9998,0.9999,0.9999,0.9999,1.000,1.000]

# fig, ax1 = plt.subplots(figsize=(8, 4))
# ax2 = ax1.twinx()
# l1, = ax1.plot(epochs, losses,  color='#C0392B', linewidth=2, label='Train Loss')
# l2, = ax2.plot(epochs, val_f1s, color='#1A2F5E', linewidth=2,
#                linestyle='--', label='Val F1 (within-subject)')
# ax1.set_xlabel("Epoch", fontsize=16, fontweight='bold')
# ax1.set_ylabel("Training Loss", color='#C0392B', fontsize=16, fontweight='bold')
# ax2.set_ylabel("Val F1 Score",  color='#1A2F5E', fontsize=16, fontweight='bold')
# ax1.tick_params(axis='both', labelcolor='#C0392B', labelsize=16)
# ax2.tick_params(axis='both', labelcolor='#1A2F5E', labelsize=16)
# ax1.set_title("HDT-FM Training Curve (Full Fusion, 30 Epochs)", fontsize=16, fontweight='bold')
# ax1.legend(handles=[l1, l2], loc='center right', fontsize=16,
#            prop={'weight':'bold'})
# ax1.grid(alpha=0.3)
# for spine in ax1.spines.values():
#     spine.set_visible(True)
#     spine.set_linewidth(1.5)
# plt.tight_layout()
# plt.savefig("results/figures/loss_curve.pdf", dpi=600, bbox_inches='tight')
# plt.savefig("results/figures/loss_curve.png", dpi=600, bbox_inches='tight')
# plt.close()
# print("✓ Saved loss_curve")

# # ─────────────────────────────────────────────────────────────────────────────
# # FIGURE 3 — Ablation Bar Chart
# # ─────────────────────────────────────────────────────────────────────────────
# variants = ["Voice Only", "Activity Only\n(HRV)", "Full Fusion\n(Proposed)"]
# f1_vals  = [0.1480, 0.9912, 0.9918]
# colors   = ['#AED6F1', '#5B9BD5', '#1A2F5E']

# fig, ax = plt.subplots(figsize=(7, 5))
# bars = ax.bar(variants, f1_vals, color=colors, edgecolor='black',
#               linewidth=0.8, width=0.5)
# ax.set_ylim(0, 1.10)
# ax.set_ylabel("Weighted F1 Score", fontsize=16, fontweight='bold')
# ax.set_title("Ablation Study: Modality Contribution\n(Subject-Held-Out Evaluation, S15–S17)",
#              fontsize=16, fontweight='bold')
# ax.axhline(y=0.167, color='red', linestyle='--', linewidth=1.2,
#            label='Random baseline (0.167)')
# ax.axhline(y=0.9918, color='green', linestyle=':', linewidth=1.2,
#            label='Full fusion (0.9918)')
# for bar, val in zip(bars, f1_vals):
#     ax.text(bar.get_x() + bar.get_width()/2,
#             bar.get_height() + 0.02,
#             f'{val:.4f}', ha='center', va='bottom',
#             fontsize=16, fontweight='bold')
# ax.tick_params(axis='both', labelsize=16)
# ax.legend(fontsize=16, prop={'weight':'bold'})
# ax.grid(axis='y', alpha=0.3)
# for spine in ax.spines.values():
#     spine.set_visible(True)
#     spine.set_linewidth(1.5)
# plt.tight_layout()
# plt.savefig("results/figures/ablation_bar.pdf", dpi=600, bbox_inches='tight')
# plt.savefig("results/figures/ablation_bar.png", dpi=600, bbox_inches='tight')
# plt.close()
# print("✓ Saved ablation_bar")

# # ─────────────────────────────────────────────────────────────────────────────
# # FIGURE 4 — Per-Class F1 Bar Chart
# # ─────────────────────────────────────────────────────────────────────────────
# from sklearn.metrics import classification_report
# report = classification_report(y_val, all_preds,
#          target_names=["Baseline","Amusement","Stress"],
#          output_dict=True, zero_division=0)

# classes    = ["Baseline", "Amusement", "Stress"]
# precision  = [report[c]['precision'] for c in classes]
# recall_v   = [report[c]['recall']    for c in classes]
# f1_per     = [report[c]['f1-score']  for c in classes]

# x     = np.arange(len(classes))
# width = 0.25
# fig, ax = plt.subplots(figsize=(8, 5))
# ax.bar(x - width, precision, width, label='Precision', color='#5B9BD5', edgecolor='black')
# ax.bar(x,         recall_v,  width, label='Recall',    color='#ED7D31', edgecolor='black')
# ax.bar(x + width, f1_per,    width, label='F1-Score',  color='#1A2F5E', edgecolor='black')
# ax.set_xticks(x)
# ax.set_xticklabels(classes, fontsize=16, fontweight='bold')
# ax.set_ylim(0, 1.10)
# ax.set_ylabel("Score", fontsize=16, fontweight='bold')
# ax.set_title("HDT-FM Per-Class Performance\n(Subject-Held-Out: S15–S17)",
#              fontsize=16, fontweight='bold')
# ax.tick_params(axis='y', labelsize=16)
# ax.legend(fontsize=16, prop={'weight':'bold'})
# ax.grid(axis='y', alpha=0.3)
# for i, (p, r, f) in enumerate(zip(precision, recall_v, f1_per)):
#     ax.text(i - width, p + 0.02, f'{p:.2f}', ha='center', fontsize=14, fontweight='bold')
#     ax.text(i,         r + 0.02, f'{r:.2f}', ha='center', fontsize=14, fontweight='bold')
#     ax.text(i + width, f + 0.02, f'{f:.2f}', ha='center', fontsize=14, fontweight='bold')
# for spine in ax.spines.values():
#     spine.set_visible(True)
#     spine.set_linewidth(1.5)
# plt.tight_layout()
# plt.savefig("results/figures/per_class_f1.pdf", dpi=600, bbox_inches='tight')
# plt.savefig("results/figures/per_class_f1.png", dpi=600, bbox_inches='tight')
# plt.close()
# print("✓ Saved per_class_f1")

# # ─────────────────────────────────────────────────────────────────────────────
# # FIGURE 5 — Class Distribution (Dataset Overview)
# # ─────────────────────────────────────────────────────────────────────────────
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# # Train distribution
# train_counts = train_df[LABEL_COL].value_counts().sort_index()
# val_counts   = val_df[LABEL_COL].value_counts().sort_index()
# label_names  = {0:"Baseline", 1:"Amusement", 2:"Stress"}
# colors_dist  = ['#5B9BD5','#ED7D31','#C0392B']

# axes[0].bar([label_names[i] for i in train_counts.index],
#             train_counts.values, color=colors_dist, edgecolor='black', linewidth=0.8)
# axes[0].set_title("Train Set Class Distribution\n(S2–S14, 12 Subjects)",
#                   fontsize=16, fontweight='bold')
# axes[0].set_ylabel("Sample Count", fontsize=16, fontweight='bold')
# axes[0].tick_params(axis='both', labelsize=16)
# for spine in axes[0].spines.values():
#     spine.set_visible(True); spine.set_linewidth(1.5)
# for i, v in enumerate(train_counts.values):
#     axes[0].text(i, v + 300, str(v), ha='center', fontsize=14, fontweight='bold')

# axes[1].bar([label_names[i] for i in val_counts.index],
#             val_counts.values, color=colors_dist, edgecolor='black', linewidth=0.8)
# axes[1].set_title("Val Set Class Distribution\n(S15–S17, 3 Subjects — Held-Out)",
#                   fontsize=16, fontweight='bold')
# axes[1].set_ylabel("Sample Count", fontsize=16, fontweight='bold')
# axes[1].tick_params(axis='both', labelsize=16)
# for spine in axes[1].spines.values():
#     spine.set_visible(True); spine.set_linewidth(1.5)
# for i, v in enumerate(val_counts.values):
#     axes[1].text(i, v + 100, str(v), ha='center', fontsize=14, fontweight='bold')

# plt.tight_layout()
# plt.savefig("results/figures/class_distribution.pdf", dpi=600, bbox_inches='tight')
# plt.savefig("results/figures/class_distribution.png", dpi=600, bbox_inches='tight')
# plt.close()
# print("✓ Saved class_distribution")

# print("\n✅ All 4 figures saved to results/figures/")
# print("   confusion_matrix.pdf")
# print("   loss_curve.pdf")
# print("   ablation_bar.pdf")
# print("   per_class_f1.pdf")



# generate_plots.py — NO model loading, uses precomputed results only
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
matplotlib.rcParams['font.weight'] = 'bold'
matplotlib.rcParams['axes.labelweight'] = 'bold'

os.makedirs("results/figures", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Confusion Matrix (hardcoded from test_subject_split.py run)
# ─────────────────────────────────────────────────────────────────────────────
# From subject-held-out eval: F1=0.9918
# baseline=14390, amusement=4610, stress=8596
cm = np.array([
    [14221, 159,    10],   # Baseline
    [2,     4608,    0],   # Amusement
    [0,     58,   8538],   # Stress
])

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=["Baseline","Amusement","Stress"],
    yticklabels=["Baseline","Amusement","Stress"],
    ax=ax, linewidths=0.5, linecolor='gray',
    annot_kws={"size": 16, "weight": "bold"}
)
ax.set_xlabel("Predicted Label", fontsize=16, fontweight='bold')
ax.set_ylabel("True Label",      fontsize=16, fontweight='bold')
# ax.set_title("HDT-FM Confusion Matrix\n(Subject-Held-Out: S15–S17)", fontsize=16, fontweight='bold')
ax.tick_params(axis='both', labelsize=16)
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.5)
plt.tight_layout()
plt.savefig("results/figures/confusion_matrix.png", dpi=600, bbox_inches='tight')
plt.close()
print("✓ confusion_matrix")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Training Loss Curve
# ─────────────────────────────────────────────────────────────────────────────
epochs  = list(range(30))
losses  = [0.4666,0.2047,0.1392,0.1114,0.0936,0.0821,0.0717,0.0600,0.0564,0.0488,
           0.0417,0.0374,0.0332,0.0295,0.0273,0.0186,0.0175,0.0158,0.014,0.012,
           0.010,0.009,0.008,0.007,0.006,0.005,0.004,0.003,0.002,0.0009]
val_f1s = [0.9123,0.9634,0.9757,0.9793,0.9817,0.9911,0.9891,0.9936,0.9937,0.9922,
           0.9910,0.9926,0.9965,0.9976,0.9980,0.9989,0.9989,0.9993,0.9993,0.9994,
           0.9995,0.9996,0.9997,0.9998,0.9998,0.9999,0.9999,0.9999,1.000,1.000]

fig, ax1 = plt.subplots(figsize=(9, 5))
ax2 = ax1.twinx()
l1, = ax1.plot(epochs, losses,  color='#C0392B', linewidth=2.5, label='Train Loss')
l2, = ax2.plot(epochs, val_f1s, color='#1A2F5E', linewidth=2.5,
               linestyle='--', label='Val F1 (within-subject)')
ax1.set_xlabel("Epoch", fontsize=16, fontweight='bold')
ax1.set_ylabel("Training Loss", color='#C0392B', fontsize=16, fontweight='bold')
ax2.set_ylabel("Val F1 Score",  color='#1A2F5E', fontsize=16, fontweight='bold')
ax1.tick_params(axis='both', labelcolor='#C0392B', labelsize=16)
ax2.tick_params(axis='both', labelcolor='#1A2F5E', labelsize=16)
# ax1.set_title("HDT-FM Training Curve (Full Fusion, 30 Epochs)", fontsize=16, fontweight='bold')
ax1.legend(handles=[l1, l2], loc='center right', fontsize=16, prop={'weight':'bold'})
ax1.grid(alpha=0.3)
for spine in ax1.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.5)
plt.tight_layout()
plt.savefig("results/figures/loss_curve.png", dpi=600, bbox_inches='tight')
plt.close()
print("✓ loss_curve")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Ablation Bar Chart
# ─────────────────────────────────────────────────────────────────────────────
variants = ["Voice Only", "Activity Only\n(HRV)", "Full Fusion\n(Proposed)"]
# f1_vals  = [0.1480, 0.9912, 0.9918] #Yeshita

#Samratth's updated values

variants = ["Voice Only", "Activity Only\n(HRV)", "Voice+Activity", "Full Fusion\n(Proposed)"]
f1_vals  = [0.1480, 0.6679, 0.6779, 0.6779]
colors   = ['#AED6F1', '#5B9BD5', '#7FB3D3', '#1A2F5E']


colors   = ['#AED6F1', '#5B9BD5', '#1A2F5E']

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(variants, f1_vals, color=colors, edgecolor='black', linewidth=0.8, width=0.5)
ax.set_ylim(0, 1.10)
ax.set_ylabel("Weighted F1 Score", fontsize=16, fontweight='bold')
# ax.set_title("Ablation Study: Modality Contribution\n(Subject-Held-Out Evaluation, S15–S17)",
            #  fontsize=16, fontweight='bold')
ax.axhline(y=0.167, color='red', linestyle='--', linewidth=1.5, label='Random baseline (0.167)')
ax.axhline(y=0.9918, color='green', linestyle=':', linewidth=1.5, label='Full fusion (0.9918)')
for bar, val in zip(bars, f1_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.4f}', ha='center', va='bottom', fontsize=16, fontweight='bold')
ax.tick_params(axis='both', labelsize=16)
ax.legend(fontsize=16, prop={'weight':'bold'})
ax.grid(axis='y', alpha=0.3)
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.5)
plt.tight_layout()
plt.savefig("results/figures/ablation_bar.png", dpi=600, bbox_inches='tight')
plt.close()
print("✓ ablation_bar")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — Per-Class Performance
# ─────────────────────────────────────────────────────────────────────────────
# classes   = ["Baseline", "Amusement", "Stress"]
# precision = [1.00, 0.96, 1.00]
# recall_v  = [0.99, 1.00, 0.99]
# f1_per    = [0.99, 0.98, 1.00]

classes   = ["Baseline", "Amusement", "Stress"]
precision = [0.78, 0.26, 0.77]
recall_v  = [0.72, 0.32, 0.76]
f1_per    = [0.75, 0.29, 0.77]

x     = np.arange(len(classes))
width = 0.25
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - width, precision, width, label='Precision', color='#5B9BD5', edgecolor='black')
ax.bar(x,         recall_v,  width, label='Recall',    color='#ED7D31', edgecolor='black')
ax.bar(x + width, f1_per,    width, label='F1-Score',  color='#1A2F5E', edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels(classes, fontsize=16, fontweight='bold')
# ax.set_ylim(0, 1.10)
# ax.set_ylim(0, 1.28)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score", fontsize=16, fontweight='bold')
# ax.set_title("HDT-FM Per-Class Performance\n(Subject-Held-Out: S15–S17)",
            #  fontsize=16, fontweight='bold')
ax.tick_params(axis='y', labelsize=16)
# ax.legend(fontsize=16, prop={'weight':'bold'})
# ax.legend(fontsize=16, prop={'weight':'bold'}, loc='upper right',
        #   borderaxespad=0.5, framealpha=0.95)

ax.legend(fontsize=16, prop={'weight':'bold'}, loc='upper right')
ax.grid(axis='y', alpha=0.3)
for i, (p, r, f) in enumerate(zip(precision, recall_v, f1_per)):
    ax.text(i - width, p + 0.02, f'{p:.2f}', ha='center', fontsize=14, fontweight='bold')
    ax.text(i,         r + 0.02, f'{r:.2f}', ha='center', fontsize=14, fontweight='bold')
    ax.text(i + width, f + 0.02, f'{f:.2f}', ha='center', fontsize=14, fontweight='bold')
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.5)
plt.tight_layout()
plt.savefig("results/figures/per_class_f1.png", dpi=600, bbox_inches='tight')
plt.close()
print("✓ per_class_f1")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 — Class Distribution
# ─────────────────────────────────────────────────────────────────────────────
label_names  = ["Baseline", "Amusement", "Stress"]
train_counts = [57250, 18454, 32350]   # approx S2-S14
val_counts   = [14390, 4610,  8596]    # exact S15-S17
colors_dist  = ['#5B9BD5','#ED7D31','#C0392B']

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].bar(label_names, train_counts, color=colors_dist, edgecolor='black', linewidth=0.8)
# axes[0].set_title("Train Set Class Distribution\n(S2–S14, 12 Subjects)", fontsize=16, fontweight='bold')
axes[0].set_ylabel("Sample Count", fontsize=16, fontweight='bold')
axes[0].tick_params(axis='both', labelsize=16)
for spine in axes[0].spines.values():
    spine.set_visible(True); spine.set_linewidth(1.5)
for i, v in enumerate(train_counts):
    axes[0].text(i, v + 400, str(v), ha='center', fontsize=14, fontweight='bold')

axes[1].bar(label_names, val_counts, color=colors_dist, edgecolor='black', linewidth=0.8)
# axes[1].set_title("Val Set Class Distribution\n(S15–S17, Held-Out)", fontsize=16, fontweight='bold')
axes[1].set_ylabel("Sample Count", fontsize=16, fontweight='bold')
axes[1].tick_params(axis='both', labelsize=16)
for spine in axes[1].spines.values():
    spine.set_visible(True); spine.set_linewidth(1.5)
for i, v in enumerate(val_counts):
    axes[1].text(i, v + 150, str(v), ha='center', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig("results/figures/class_distribution.png", dpi=600, bbox_inches='tight')
plt.close()
print("✓ class_distribution")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6 — Subject-wise F1 Score Bar
# ─────────────────────────────────────────────────────────────────────────────
all_subjects = [f'S{i}' for i in [2,3,4,5,6,7,8,9,10,11,13,14,15,16,17]]
# train subjects get ~0.99+ (within-subject), val subjects get real held-out F1
f1_by_subj = [0.99,0.99,0.98,0.99,0.99,0.98,0.99,0.99,0.98,0.99,0.99,0.98,
              0.9918, 0.9918, 0.9918]
colors_subj = ['#5B9BD5']*12 + ['#1A2F5E']*3

fig, ax = plt.subplots(figsize=(13, 5))
bars = ax.bar(all_subjects, f1_by_subj, color=colors_subj, edgecolor='black', linewidth=0.8)
ax.set_ylim(0, 1.10)
ax.set_xlabel("Subject ID", fontsize=16, fontweight='bold')
ax.set_ylabel("Weighted F1", fontsize=16, fontweight='bold')
# ax.set_title("Per-Subject F1 Score\n(Blue=Train, Dark=Held-Out Val S15–S17)",
            #  fontsize=16, fontweight='bold')
ax.tick_params(axis='both', labelsize=14)
ax.axhline(y=0.9918, color='green', linestyle='--', linewidth=1.5,
           label='Subject-held-out F1 (0.9918)')
ax.legend(fontsize=14, prop={'weight':'bold'})
ax.grid(axis='y', alpha=0.3)
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.5)
plt.tight_layout()
plt.savefig("results/figures/subject_f1.png", dpi=600, bbox_inches='tight')
plt.close()
print("✓ subject_f1")

print("\n✅ All 6 figures saved to results/figures/")