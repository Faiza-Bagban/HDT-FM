# scripts/full_eval_for_paper.py
"""
Extracts everything Yeshita needs:
1. Confusion matrix (3x3)
2. Training loss per epoch (from run output — hardcoded from v3)
3. Val F1 per epoch (from run output — hardcoded from v3)
4. Subject-wise F1
5. Best model path
6. AUC-ROC (weighted)
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (f1_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from models.hdt_model import HDTModel

FEATURE_COLS = [
    'MEAN_RR','MEDIAN_RR','SDRR','RMSSD','SDSD','HR',
    'pNN25','pNN50','SD1','SD2','KURT','SKEW',
    'VLF','LF','HF','TP','LF_HF','HF_LF','LF_NU','HF_NU',
]
LABEL_COL = 'condition label'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BEST_MODEL = "runs/best_model.pt"

# -- Load data -------------------------------------------------------------
df = pd.read_csv("data/WESAD/wesad-chest-combined-classification-hrv.csv")
df = df.dropna(subset=FEATURE_COLS + [LABEL_COL, 'subject id'])
subjects  = sorted(df['subject id'].unique())
val_subj  = subjects[-3:]  # [15, 16, 17]
train_df  = df[~df['subject id'].isin(val_subj)]
val_df    = df[df['subject id'].isin(val_subj)]

scaler = StandardScaler()
scaler.fit(train_df[FEATURE_COLS].values.astype(np.float32))
X_val     = scaler.transform(val_df[FEATURE_COLS].values.astype(np.float32))
X_val_pad = np.zeros((len(X_val), 128), dtype=np.float32)
X_val_pad[:, :X_val.shape[1]] = X_val
y_val = val_df[LABEL_COL].values.astype(np.int64)
subj_val = val_df['subject id'].values

# Voice embeddings
voice_embs  = np.load("data/embeddings/voice_trained.npy")
n_repeat    = (len(X_val_pad) // len(voice_embs)) + 1
voice_tiled = np.tile(voice_embs, (n_repeat, 1))[:len(X_val_pad)]

# -- Load model -------------------------------------------------------------
model = HDTModel(n_classes=3).to(DEVICE)
state = torch.load(BEST_MODEL, map_location=DEVICE)
if "model" in state:
    state = state["model"]
model.load_state_dict(state, strict=False)
model.eval()

# -- Run full eval -----------------------------------------------------------
all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for i in range(0, len(X_val_pad), 64):
        end = min(i + 64, len(X_val_pad))
        activity  = torch.tensor(X_val_pad[i:end]).to(DEVICE)
        mobility  = torch.zeros(end - i, 64).to(DEVICE)
        voice_emb = torch.tensor(voice_tiled[i:end]).float().to(DEVICE)

        out = model(voice_emb=voice_emb, activity=activity, mobility=mobility)
        logits = out[0]
        probs = torch.softmax(logits, dim=1)

        all_preds.extend(logits.argmax(-1).cpu().tolist())
        all_labels.extend(y_val[i:end].tolist())
        all_probs.append(probs.cpu().numpy())

all_probs = np.concatenate(all_probs)

# ===================================================================
# 1. CONFUSION MATRIX
# ===================================================================
print("=" * 60)
print("1. CONFUSION MATRIX (rows=true, cols=predicted)")
print("   [baseline, amusement, stress]")
print("=" * 60)
cm = confusion_matrix(all_labels, all_preds)
print(cm)
print()

# ===================================================================
# 2. TRAINING LOSS PER EPOCH (from v3 real-voice run)
# ===================================================================
print("=" * 60)
print("2. TRAINING LOSS (all 30 epochs)")
print("=" * 60)
train_losses = [
    0.4649, 0.2232, 0.1556, 0.1287, 0.1146, 0.1078, 0.0994,
    0.0921, 0.0960, 0.0865, 0.0848, 0.0815, 0.0760, 0.0739,
    0.0757, 0.0913, 0.0704, 0.0612, 0.0540, 0.0444, 0.0404,
    0.0369, 0.0310, 0.0224, 0.0219, 0.0197, 0.0185, 0.0160,
    0.0149, 0.0139,
]
for i, l in enumerate(train_losses):
    print(f"  Epoch {i:02d}: {l:.4f}")
print()

# ===================================================================
# 3. VAL F1 PER EPOCH
# ===================================================================
print("=" * 60)
print("3. VAL F1 (all 30 epochs)")
print("=" * 60)
val_f1s = [
    0.5990, 0.5579, 0.6300, 0.5597, 0.6489, 0.6244, 0.6511,
    0.6365, 0.6624, 0.6527, 0.6623, 0.6761, 0.6635, 0.6697,
    0.6364, 0.6312, 0.6385, 0.6335, 0.6168, 0.6056, 0.6137,
    0.6227, 0.6329, 0.6307, 0.6340, 0.6300, 0.6371, 0.6439,
    0.6389, 0.6382,
]
for i, f in enumerate(val_f1s):
    print(f"  Epoch {i:02d}: {f:.4f}")
print(f"\n  Best: {max(val_f1s):.4f} (epoch {val_f1s.index(max(val_f1s))})")
print()

# ===================================================================
# 4. SUBJECT-WISE F1
# ===================================================================
print("=" * 60)
print("4. SUBJECT-WISE F1")
print("=" * 60)
for s in sorted(set(subj_val)):
    mask = subj_val == s
    s_preds = [all_preds[j] for j in range(len(all_preds)) if mask[j]]
    s_labels = [all_labels[j] for j in range(len(all_labels)) if mask[j]]
    s_f1 = f1_score(s_labels, s_preds, average="weighted", zero_division=0)
    n = sum(mask)
    print(f"  Subject {int(s):2d}: F1={s_f1:.4f}  (n={n})")
print()

# ===================================================================
# 5. BEST MODEL PATH
# ===================================================================
print("=" * 60)
print("5. BEST MODEL PATH")
print("=" * 60)
print(f"  {os.path.abspath(BEST_MODEL)}")
print(f"  Size: {os.path.getsize(BEST_MODEL) / 1e6:.1f} MB")
print()

# ===================================================================
# 6. AUC-ROC (weighted, one-vs-rest)
# ===================================================================
print("=" * 60)
print("6. AUC-ROC")
print("=" * 60)
try:
    auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="weighted")
    print(f"  Weighted AUC-ROC: {auc:.4f}")
except ValueError as e:
    print(f"  Could not compute AUC: {e}")
print()

# ===================================================================
# FULL CLASSIFICATION REPORT
# ===================================================================
print("=" * 60)
print("FULL CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(all_labels, all_preds,
      target_names=["baseline", "amusement", "stress"], zero_division=0))