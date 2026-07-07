# scripts/extract_voice_embeddings_trained.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch, torchaudio, numpy as np
from pathlib import Path
from tqdm import tqdm

# import VoiceClassifier from train_voice_encoder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
from train_voice_encoder import VoiceClassifier

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SUBSET_DIR = Path("data/VOICE_subset")
TARGET_SR  = 16000

model = VoiceClassifier(n_classes=6).to(DEVICE)
model.load_state_dict(torch.load("runs_voice/best_voice_encoder.pt", map_location=DEVICE))
model.eval()

embeddings = []
os.makedirs("data/embeddings", exist_ok=True)

with torch.no_grad():
    for f in tqdm(sorted(SUBSET_DIR.glob("*.wav"))):
        wav, sr = torchaudio.load(str(f))
        if sr != TARGET_SR:
            wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
        wav = wav.mean(0)
        if len(wav) > TARGET_SR:
            wav = wav[:TARGET_SR]
        else:
            wav = torch.nn.functional.pad(wav, (0, TARGET_SR - len(wav)))

        wav = wav.unsqueeze(0).to(DEVICE)
        with torch.amp.autocast('cuda', enabled=(DEVICE=="cuda")):
            _, proj = model(wav)              # (1, 256)
        embeddings.append(proj.squeeze(0).cpu().numpy())

emb_array = np.array(embeddings)
np.save("data/embeddings/voice_trained.npy", emb_array)
print(f"Saved: {emb_array.shape}")   # expect (240, 256)