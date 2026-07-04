# scripts/run_ablation_fast.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import pandas as pd
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

df = pd.read_csv("data/WESAD/wesad-chest-combined-classification-hrv.csv")
df = df.dropna(subset=FEATURE_COLS + [LABEL_COL, 'subject id'])
subjects = sorted(df['subject id'].unique())
val_subj = subjects[-3:]
train_subj = [s for s in subjects if s not in val_subj]

train_df = df[df['subject id'].isin(train_subj)]
val_df   = df[df['subject id'].isin(val_subj)]

scaler  = StandardScaler()
X_train = scaler.fit_transform(train_df[FEATURE_COLS].values.astype(np.float32))
X_val   = scaler.transform(val_df[FEATURE_COLS].values.astype(np.float32))

def pad128(X):
    out = np.zeros((len(X), 128), dtype=np.float32)
    out[:, :X.shape[1]] = X
    return out

X_val_pad = pad128(X_val)
y_val = val_df[LABEL_COL].values.astype(np.int64)

VARIANTS = [
    ("voice_only",    True,  False, False),
    ("activity_only", False, True,  False),
    ("full_fusion",   True,  True,  False),
]

results = []
for name, use_voice, use_activity, use_mobility in VARIANTS:
    model = HDTModel(n_classes=3).to(DEVICE)
    model.load_state_dict(torch.load("runs/best_model.pt", map_location=DEVICE))
    model.eval()

    all_preds, all_labels = [], []
    batch_size = 64

    with torch.no_grad():
        for i in range(0, len(X_val_pad), batch_size):
            activity = torch.tensor(X_val_pad[i:i+batch_size]).to(DEVICE)
            mobility = torch.zeros(len(activity), 64).to(DEVICE)
            audio    = torch.randn(len(activity), 16000).to(DEVICE)

            if not use_voice:    audio    = torch.zeros_like(audio)
            if not use_activity: activity = torch.zeros_like(activity)
            if not use_mobility: mobility = torch.zeros_like(mobility)

            logits, _ = model(audio, activity, mobility)
            preds = logits.argmax(-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y_val[i:i+batch_size].tolist())

    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    print(f"\n{'='*40}")
    print(f"Variant: {name} | F1={f1:.4f}")
    print(classification_report(all_labels, all_preds,
          target_names=["baseline","amusement","stress"], zero_division=0))
    results.append({"variant": name, "val_f1": round(f1, 4)})

import pandas as pd
os.makedirs("results", exist_ok=True)
pd.DataFrame(results).to_csv("results/ablation.csv", index=False)
print("\nSaved: results/ablation.csv")
print(pd.DataFrame(results))