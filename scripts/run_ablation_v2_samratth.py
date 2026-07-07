# scripts/run_ablation_v2_samratth.py
"""
Ablation with REAL CREMA-D voice embeddings.
Uses Samratth's HDTModel (4 outputs) + our trained checkpoint.
"""
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

# -- Load WESAD val set ---------------------------------------------------
df = pd.read_csv("data/WESAD/wesad-chest-combined-classification-hrv.csv")
df = df.dropna(subset=FEATURE_COLS + [LABEL_COL, 'subject id'])
subjects  = sorted(df['subject id'].unique())
val_subj  = subjects[-3:]
train_df  = df[~df['subject id'].isin(val_subj)]
val_df    = df[df['subject id'].isin(val_subj)]

scaler = StandardScaler()
scaler.fit(train_df[FEATURE_COLS].values.astype(np.float32))
X_val     = scaler.transform(val_df[FEATURE_COLS].values.astype(np.float32))
X_val_pad = np.zeros((len(X_val), 128), dtype=np.float32)
X_val_pad[:, :X_val.shape[1]] = X_val
y_val = val_df[LABEL_COL].values.astype(np.int64)

# -- Load + tile voice embeddings -----------------------------------------
voice_embs  = np.load("data/embeddings/voice_trained.npy")   # (240, 256)
n_repeat    = (len(X_val_pad) // len(voice_embs)) + 1
voice_tiled = np.tile(voice_embs, (n_repeat, 1))[:len(X_val_pad)]

print(f"Val samples: {len(X_val_pad)}")
print(f"Voice embeddings: {voice_embs.shape} -> tiled to {voice_tiled.shape}")

# -- Load model ------------------------------------------------------------
model = HDTModel(n_classes=3).to(DEVICE)

# Try loading checkpoint
ckpt_path = "runs/best_model.pt"
if not os.path.isfile(ckpt_path):
    # Fallback: find latest epoch checkpoint
    ckpts = sorted([f for f in os.listdir("runs") if f.startswith("ckpt_epoch") and f.endswith(".pt")])
    if ckpts:
        ckpt_path = os.path.join("runs", ckpts[-1])
        print(f"best_model.pt not found, using {ckpt_path}")
    else:
        print("ERROR: No checkpoints found in runs/. Train first.")
        sys.exit(1)

state = torch.load(ckpt_path, map_location=DEVICE)
# Handle both raw state_dict and wrapped checkpoint
if "model" in state:
    state = state["model"]
model.load_state_dict(state, strict=False)
model.eval()
print(f"Loaded checkpoint: {ckpt_path}")

# -- Ablation variants -----------------------------------------------------
VARIANTS = [
    ("voice-only",         True,  False, False),
    ("activity-only",      False, True,  False),
    ("mobility-only",      False, False, True),
    ("voice+activity",     True,  True,  False),
    ("voice+mobility",     True,  False, True),
    ("activity+mobility",  False, True,  True),
    ("full-fusion",        True,  True,  True),
]

results = []
for name, use_voice, use_activity, use_mobility in VARIANTS:
    all_preds, all_labels = [], []
    with torch.no_grad():
        for i in range(0, len(X_val_pad), 64):
            end = min(i + 64, len(X_val_pad))
            activity  = torch.tensor(X_val_pad[i:end]).to(DEVICE)
            mobility  = torch.zeros(end - i, 64).to(DEVICE)
            voice_emb = torch.tensor(voice_tiled[i:end]).float().to(DEVICE)

            if not use_voice:    voice_emb = torch.zeros_like(voice_emb)
            if not use_activity: activity  = torch.zeros_like(activity)
            if not use_mobility: mobility  = torch.zeros_like(mobility)

            # Our model returns 4 outputs
            out = model(voice_emb=voice_emb, activity=activity, mobility=mobility)
            logits = out[0]

            all_preds.extend(logits.argmax(-1).cpu().tolist())
            all_labels.extend(y_val[i:end].tolist())

    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    print(f"\n{'='*50}")
    print(f"Variant: {name} | F1={f1:.4f}")
    print(classification_report(all_labels, all_preds,
          target_names=["baseline","amusement","stress"], zero_division=0))
    results.append({"variant": name, "val_f1": round(f1, 4)})

# -- Save + print -----------------------------------------------------------
os.makedirs("results", exist_ok=True)
pd.DataFrame(results).to_csv("results/ablation_v2_samratth.csv", index=False)

print(f"\n{'='*50}")
print("FINAL ABLATION TABLE (with real voice embeddings)")
print('='*50)
for r in results:
    bar = '#' * int(r["val_f1"] * 40)
    print(f"  {r['variant']:<20s} F1={r['val_f1']:.4f}  {bar}")
print(f"\nSaved -> results/ablation_v2_samratth.csv")