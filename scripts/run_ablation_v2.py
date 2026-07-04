# scripts/run_ablation_v2.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch, numpy as np, pandas as pd
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from models.hdt_model import HDTModel

FEATURE_COLS = [
    'MEAN_RR','MEDIAN_RR','SDRR','RMSSD','SDSD','HR',
    'pNN25','pNN50','SD1','SD2','KURT','SKEW',
    'VLF','LF','HF','TP','LF_HF','HF_LF','LF_NU','HF_NU',
]
LABEL_COL = 'condition label'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load WESAD val set ────────────────────────────────────────────────────────
df = pd.read_csv("data/WESAD/wesad-chest-combined-classification-hrv.csv")
df = df.dropna(subset=FEATURE_COLS + [LABEL_COL, 'subject id'])
subjects  = sorted(df['subject id'].unique())
val_subj  = subjects[-3:]
train_df  = df[~df['subject id'].isin(val_subj)]
val_df    = df[df['subject id'].isin(val_subj)]

scaler   = StandardScaler()
scaler.fit(train_df[FEATURE_COLS].values.astype(np.float32))
X_val    = scaler.transform(val_df[FEATURE_COLS].values.astype(np.float32))
X_val_pad = np.zeros((len(X_val), 128), dtype=np.float32)
X_val_pad[:, :X_val.shape[1]] = X_val
y_val = val_df[LABEL_COL].values.astype(np.int64)

# ── Load + tile voice embeddings ──────────────────────────────────────────────
voice_embs = np.load("data/embeddings/voice_trained.npy")   # (240, 256)
n_repeat   = (len(X_val_pad) // len(voice_embs)) + 1
voice_tiled = np.tile(voice_embs, (n_repeat, 1))[:len(X_val_pad)]  # (N_val, 256)

# ── Load model ────────────────────────────────────────────────────────────────
model = HDTModel(n_classes=3).to(DEVICE)
model.load_state_dict(torch.load("runs/best_model.pt", map_location=DEVICE))
model.eval()

# ── Ablation variants ─────────────────────────────────────────────────────────
VARIANTS = [
    ("voice_only",    True,  False),
    ("activity_only", False, True),
    ("full_fusion",   True,  True),
]

results = []
for name, use_voice, use_activity in VARIANTS:
    all_preds, all_labels = [], []
    with torch.no_grad():
        for i in range(0, len(X_val_pad), 64):
            activity  = torch.tensor(X_val_pad[i:i+64]).to(DEVICE)
            mobility  = torch.zeros(len(activity), 64).to(DEVICE)
            voice_emb = torch.tensor(voice_tiled[i:i+64]).float().to(DEVICE)

            if not use_voice:    voice_emb = torch.zeros_like(voice_emb)
            if not use_activity: activity  = torch.zeros_like(activity)

            logits, _ = model(voice_emb=voice_emb,
                              activity=activity,
                              mobility=mobility)
            all_preds.extend(logits.argmax(-1).cpu().tolist())
            all_labels.extend(y_val[i:i+64].tolist())

    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    print(f"\n{'='*40}")
    print(f"Variant: {name} | F1={f1:.4f}")
    print(classification_report(all_labels, all_preds,
          target_names=["baseline","amusement","stress"], zero_division=0))
    results.append({"variant": name, "val_f1": round(f1, 4)})

os.makedirs("results", exist_ok=True)
pd.DataFrame(results).to_csv("results/ablation_v2.csv", index=False)
print("\n=== FINAL ABLATION TABLE ===")
print(pd.DataFrame(results).to_string(index=False))