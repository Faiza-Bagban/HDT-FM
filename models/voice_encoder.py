# models/voice_encoder.py
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class VoiceEncoder(nn.Module):
    def __init__(self, out_dim=256, freeze_backbone=True):
        super().__init__()
        self.backbone = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.proj = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, input_values, attention_mask=None):
        with torch.no_grad():
            hidden = self.backbone(
                input_values, attention_mask=attention_mask
            ).last_hidden_state  # (B, T, 768)
        pooled = hidden.mean(dim=1)        # (B, 768)
        return self.proj(pooled)           # (B, 256)