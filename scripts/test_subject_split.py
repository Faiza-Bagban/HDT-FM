import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# scripts/test_subject_split.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
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
print(f"Subjects: {subjects}")

# leave-one-subject-out: train on all except last 3, val on last 3
train_subj = subjects[:-3]
val_subj   = subjects[-3:]
print(f"Train subjects: {train_subj}")
print(f"Val subjects:   {val_subj}")

train_df = df[df['subject id'].isin(train_subj)]
val_df   = df[df['subject id'].isin(val_subj)]

scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[FEATURE_COLS].values.astype(np.float32))
X_val   = scaler.transform(val_df[FEATURE_COLS].values.astype(np.float32))

def pad128(X):
    out = np.zeros((len(X), 128), dtype=np.float32)
    out[:, :X.shape[1]] = X
    return out

X_train = pad128(X_train)
X_val   = pad128(X_val)
y_train = train_df[LABEL_COL].values.astype(np.int64)
y_val   = val_df[LABEL_COL].values.astype(np.int64)

print(f"Train: {len(X_train)} | Val: {len(X_val)}")

# quick eval: load best model, test on subject-held-out val
model = HDTModel(n_classes=3).to(DEVICE)
model.load_state_dict(torch.load("runs/best_model.pt", map_location=DEVICE))
model.eval()

all_preds, all_labels = [], []
batch_size = 64
with torch.no_grad():
    for i in range(0, len(X_val), batch_size):
        activity = torch.tensor(X_val[i:i+batch_size]).to(DEVICE)
        mobility = torch.zeros(len(activity), 64).to(DEVICE)
        audio    = torch.randn(len(activity), 16000).to(DEVICE)
        logits, _ = model(audio, activity, mobility)
        preds = logits.argmax(-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y_val[i:i+batch_size].tolist())

f1 = f1_score(all_labels, all_preds, average="weighted")
print(f"\nSubject-held-out val F1: {f1:.4f}")
