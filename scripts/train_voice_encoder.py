# scripts/train_voice_encoder.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import Wav2Vec2Model
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SUBSET_DIR = Path("data/VOICE_subset")
TARGET_SR  = 16000
N_EMOTIONS = 6
EMOTION_MAP = {"ANG":0, "DIS":1, "FEA":2, "HAP":3, "NEU":4, "SAD":5}

# ── Dataset ───────────────────────────────────────────────────────────────────
class VoiceDataset(Dataset):
    def __init__(self, subset_dir, max_len=16000):
        self.samples = []
        self.max_len = max_len
        for f in sorted(Path(subset_dir).glob("*.wav")):
            parts = f.stem.split("_")
            if len(parts) < 3:
                continue
            emo = EMOTION_MAP.get(parts[2], -1)
            if emo == -1:
                continue
            self.samples.append((str(f), emo))
        print(f"VoiceDataset: {len(self.samples)} samples, {N_EMOTIONS} classes")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        wav, sr = torchaudio.load(path)
        if sr != TARGET_SR:
            wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
        wav = wav.mean(0)
        if len(wav) > self.max_len:
            wav = wav[:self.max_len]
        else:
            wav = torch.nn.functional.pad(wav, (0, self.max_len - len(wav)))
        return wav, label

# ── Model ─────────────────────────────────────────────────────────────────────
class VoiceClassifier(nn.Module):
    def __init__(self, n_classes=N_EMOTIONS, proj_dim=256):
        super().__init__()
        self.backbone = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        # freeze all except last 2 transformer layers
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.backbone.encoder.layers[-2:].parameters():
            p.requires_grad = True

        self.proj = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, proj_dim),
            nn.LayerNorm(proj_dim),
        )
        self.classifier = nn.Linear(proj_dim, n_classes)

    def forward(self, input_values):
        hidden  = self.backbone(input_values).last_hidden_state  # (B,T,768)
        pooled  = hidden.mean(dim=1)                             # (B,768)
        proj    = self.proj(pooled)                              # (B,256)
        logits  = self.classifier(proj)                          # (B,6)
        return logits, proj

# ── Train ─────────────────────────────────────────────────────────────────────
def main():
    dataset = VoiceDataset(SUBSET_DIR)
    n_val   = max(1, int(0.2 * len(dataset)))
    train_ds, val_ds = random_split(dataset, [len(dataset)-n_val, n_val])

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=0)

    model     = VoiceClassifier().to(DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {DEVICE} | Trainable params: {trainable:,}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=1e-4
    )
    ce_loss = nn.CrossEntropyLoss()
    scaler  = torch.amp.GradScaler('cuda', enabled=(DEVICE=="cuda"))
    os.makedirs("runs_voice", exist_ok=True)
    best_f1 = 0.0

    for epoch in range(20):
        model.train()
        total_loss = 0
        for wavs, labels in tqdm(train_loader, desc=f"Epoch {epoch:02d}"):
            wavs, labels = wavs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(DEVICE=="cuda")):
                logits, _ = model(wavs)
                loss = ce_loss(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for wavs, labels in val_loader:
                wavs = wavs.to(DEVICE)
                with torch.amp.autocast('cuda', enabled=(DEVICE=="cuda")):
                    logits, _ = model(wavs)
                all_preds.extend(logits.argmax(-1).cpu().tolist())
                all_labels.extend(labels.tolist())

        val_f1   = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch:02d} | loss={avg_loss:.4f} | val_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "runs_voice/best_voice_encoder.pt")
            print(f"  ★ Saved (val_f1={val_f1:.4f})")

    print(f"\nDone. Best F1: {best_f1:.4f}")

if __name__ == "__main__":
    main()