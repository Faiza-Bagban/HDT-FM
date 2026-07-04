# models/hdt_model.py
import torch
import torch.nn as nn
from models.voice_encoder import VoiceEncoder
from models.fusion import CrossModalFusion

class HDTModel(nn.Module):
    def __init__(self, activity_dim=128, mobility_dim=64,
                 fusion_dim=256, n_classes=3):
        super().__init__()
        self.voice_enc = VoiceEncoder(out_dim=fusion_dim)
        self.fusion    = CrossModalFusion(
            voice_dim=fusion_dim,
            activity_dim=activity_dim,
            mobility_dim=mobility_dim,
            out_dim=fusion_dim,
        )
        # Risk head
        self.risk_head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, audio=None, activity=None, mobility=None, voice_emb=None):
        if voice_emb is not None:
            v = voice_emb
        else:
            v = self.voice_enc(audio)
        rep = self.fusion(v, activity, mobility)
        return self.risk_head(rep), rep