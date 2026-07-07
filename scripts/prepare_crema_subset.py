# scripts/prepare_crema_subset.py
import os, random, shutil
from pathlib import Path
from collections import defaultdict

src = Path("data/CREMA")
dst = Path("data/VOICE_subset")
dst.mkdir(exist_ok=True)

files = list(src.glob("*.wav"))
print(f"Total files: {len(files)}")

EMOTION_MAP = {"ANG":0, "DIS":1, "FEA":2, "HAP":3, "NEU":4, "SAD":5}

by_emotion = defaultdict(list)
for f in files:
    parts = f.stem.split("_")
    if len(parts) >= 3 and parts[2] in EMOTION_MAP:
        by_emotion[parts[2]].append(f)

print(f"Emotions: { {k: len(v) for k,v in by_emotion.items()} }")

total = 0
for emo, fs in by_emotion.items():
    chosen = random.sample(fs, min(40, len(fs)))  # 40 per emotion
    for f in chosen:
        shutil.copy(f, dst / f.name)
    total += len(chosen)

print(f"Subset saved: {total} files → {dst}")