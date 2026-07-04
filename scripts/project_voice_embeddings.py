# scripts/project_voice_embeddings.py
import torch, numpy as np
from models.voice_encoder import VoiceEncoder

raw   = torch.tensor(np.load("data/embeddings/voice_raw.npy")).float()  # (200, 768)
proj  = torch.nn.Sequential(
    torch.nn.Linear(768, 512),
    torch.nn.LayerNorm(512),
    torch.nn.GELU(),
    torch.nn.Linear(512, 256),
    torch.nn.LayerNorm(256),
).cuda()

# NOTE: projection weights random here — they'll be trained during fusion training
# This step just verifies shapes. Actual projected embeddings come from trained model.
with torch.no_grad():
    out = proj(raw.cuda())
print(out.shape)  # (200, 256) ✓