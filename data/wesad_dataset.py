# data/wesad_dataset.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

# columns to use as activity features (HRV-based, chest sensor)
FEATURE_COLS = [
    'MEAN_RR', 'MEDIAN_RR', 'SDRR', 'RMSSD', 'SDSD', 'HR',
    'pNN25', 'pNN50', 'SD1', 'SD2', 'KURT', 'SKEW',
    'VLF', 'LF', 'HF', 'TP', 'LF_HF', 'HF_LF',
    'LF_NU', 'HF_NU',
]
LABEL_COL = 'condition label'   # 0=baseline, 1=amusement, 2=stress

class WESADDataset(Dataset):
    def __init__(self, csv_path="data/WESAD/wesad-chest-combined-classification-hrv.csv",
                 fit_scaler=True, scaler=None):

        df = pd.read_csv(csv_path)
        df = df.dropna(subset=FEATURE_COLS + [LABEL_COL])

        # use only cols that exist in file
        feat_cols = [c for c in FEATURE_COLS if c in df.columns]

        X = df[feat_cols].values.astype(np.float32)   # (N, 20)
        y = df[LABEL_COL].values.astype(np.int64)     # (N,)

        # standardise
        if fit_scaler:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        else:
            self.scaler = scaler
            X = scaler.transform(X)

        n_feat = X.shape[1]   # 20 features from CSV

        # pad to 128-d (fusion expects 128-d activity vector)
        X_pad = np.zeros((len(X), 128), dtype=np.float32)
        X_pad[:, :n_feat] = X

        self.X       = torch.tensor(X_pad)          # (N, 128)
        self.y       = torch.tensor(y)               # (N,)
        self.n_feat  = n_feat

        print(f"WESADDataset loaded: {len(self.X)} samples | "
              f"features={n_feat} (padded to 128) | "
              f"classes={dict(zip(*np.unique(y, return_counts=True)))}")

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        return {
            "activity": self.X[idx],                          # (128,)
            "mobility": torch.zeros(64, dtype=torch.float32), # placeholder
            "label":    self.y[idx],                          # scalar
        }
    # Add this function at bottom of wesad_dataset.py
def get_subject_split_loaders(
    csv_path="data/WESAD/wesad-chest-combined-classification-hrv.csv",
    batch_size=16, val_subjects=None
):
    import pandas as pd
    from torch.utils.data import DataLoader, TensorDataset

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=FEATURE_COLS + [LABEL_COL, 'subject id'])

    subjects = sorted(df['subject id'].unique())
    if val_subjects is None:
        val_subjects = subjects[-3:]   # hold out last 3 subjects
    train_subjects = [s for s in subjects if s not in val_subjects]

    print(f"Train subjects: {train_subjects}")
    print(f"Val subjects:   {val_subjects}")

    train_df = df[df['subject id'].isin(train_subjects)]
    val_df   = df[df['subject id'].isin(val_subjects)]

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(train_df[FEATURE_COLS].values.astype(np.float32))
    X_val   = scaler.transform(val_df[FEATURE_COLS].values.astype(np.float32))

    def pad128(X):
        out = np.zeros((len(X), 128), dtype=np.float32)
        out[:, :X.shape[1]] = X
        return out

    X_train, X_val = pad128(X_train), pad128(X_val)
    y_train = train_df[LABEL_COL].values.astype(np.int64)
    y_val   = val_df[LABEL_COL].values.astype(np.int64)

    mob = np.zeros((len(X_train), 64), dtype=np.float32)
    mob_v = np.zeros((len(X_val), 64), dtype=np.float32)

    train_ds = TensorDataset(
        torch.tensor(X_train), torch.tensor(mob), torch.tensor(y_train)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val), torch.tensor(mob_v), torch.tensor(y_val)
    )

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2),
        scaler,
    )