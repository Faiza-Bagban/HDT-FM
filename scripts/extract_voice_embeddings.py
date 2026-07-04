# scripts/extract_voice_embeddings.py
import torch, torchaudio, numpy as np
from pathlib import Path
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUBSET_DIR = Path("data/RAVDESS_subset")
OUT_NPY    = Path("data/embeddings/voice_raw.npy")
OUT_LABELS = Path("data/embeddings/voice_labels.npy")
OUT_NPY.parent.mkdir(parents=True, exist_ok=True)

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model     = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(DEVICE)
model.eval()  # frozen — no gradient needed

TARGET_SR = 16000
embeddings, labels = [], []

with torch.no_grad():
    for f in tqdm(sorted(SUBSET_DIR.glob("*.wav"))):
        wav, sr = torchaudio.load(f)
        if sr != TARGET_SR:
            wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
        wav = wav.mean(0)  # mono

        inputs = processor(wav, sampling_rate=TARGET_SR,
                           return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        out = model(**inputs).last_hidden_state  # (1, T, 768)
        emb = out.mean(dim=1).squeeze(0).cpu().numpy()  # (768,)
        embeddings.append(emb)

        emo = int(f.stem.split("-")[2]) - 1  # 0-indexed
        labels.append(emo)

np.save(OUT_NPY, np.array(embeddings))    # (200, 768)
np.save(OUT_LABELS, np.array(labels))
print("Saved:", OUT_NPY, OUT_LABELS)