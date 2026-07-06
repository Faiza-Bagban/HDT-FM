# models/hdt_model.py
"""
HDT-FM: Full model — Mark 50 edition.

Yeshita built:   VoiceEncoder, CrossModalFusion, RiskHead
Samratth added:  ActivityEncoder, MobilityEncoder, TemporalTransformer, TwinSimulator

Signal flow:
  raw HRV (20-d) ─→ ActivityEncoder ─→ 128-d ──┐
  raw audio      ─→ VoiceEncoder    ─→ 256-d ──┤─→ CrossModalFusion ─→ 256-d
  GPS (T, 4)     ─→ MobilityEncoder ─→  64-d ──┘         │
                                                           ▼
                                              TemporalTransformer (window)
                                                           │
                                              ┌────────────┼────────────┐
                                              ▼            ▼            ▼
                                          RiskHead    TwinSimulator  (aux loss)
                                          (3-class)   (next state)
"""

import torch
import torch.nn as nn
from models.voice_encoder import VoiceEncoder
from models.fusion import CrossModalFusion
from models.activity_encoder import ActivityEncoder
from models.mobility_encoder import MobilityEncoder
from models.temporal_transformer import TemporalTransformer
from models.twin_simulator import TwinSimulator


class HDTModel(nn.Module):
    def __init__(self, activity_raw_dim=20, activity_dim=128, mobility_dim=64,
                 fusion_dim=256, n_classes=3, use_encoders=True):
        super().__init__()
        self.use_encoders = use_encoders

        # --- Yeshita's components (untouched) ---
        self.voice_enc = VoiceEncoder(out_dim=fusion_dim)
        self.fusion = CrossModalFusion(
            voice_dim=fusion_dim,
            activity_dim=activity_dim,
            mobility_dim=mobility_dim,
            out_dim=fusion_dim,
        )

        # --- Samratth's components ---
        if use_encoders:
            self.activity_enc = ActivityEncoder(
                in_dim=activity_raw_dim, out_dim=activity_dim
            )
            self.mobility_enc = MobilityEncoder(
                input_dim=4, out_dim=mobility_dim
            )

        self.temporal = TemporalTransformer(
            embed_dim=fusion_dim, num_heads=8, num_layers=2
        )
        self.twin_sim = TwinSimulator(
            embed_dim=fusion_dim, hidden_dim=512
        )

        # Risk head (kept from Yeshita)
        self.risk_head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, audio=None, activity=None, mobility=None,
                voice_emb=None, window_mode=False):
        """
        Single-step mode (window_mode=False):
            audio:     (B, T_audio) raw waveform or None
            activity:  (B, 20) raw HRV features  [if use_encoders]
                       (B, 128) pre-padded        [if not use_encoders]
            mobility:  (B, T_gps, 4) GPS traces   [if use_encoders]
                       (B, 64) placeholder         [if not use_encoders]
            voice_emb: (B, 256) pre-extracted — skips voice_enc if given

        Returns:
            logits:     (B, 3)   risk classification
            fused:      (B, 256) fused representation
            next_state: (B, 256) twin-predicted next state
        """
        # --- Voice ---
        if voice_emb is not None:
            v = voice_emb
        else:
            v = self.voice_enc(audio)

        # --- Activity ---
        if self.use_encoders and activity.shape[-1] != 128:
            a = self.activity_enc(activity)   # (B, 20) → (B, 128)
        else:
            a = activity                      # (B, 128) already padded

        # --- Mobility ---
        if self.use_encoders and mobility.dim() == 3:
            m = self.mobility_enc(mobility)   # (B, T, 4) → (B, 64)
        else:
            m = mobility                      # (B, 64) placeholder

        # --- Fusion (Yeshita's) ---
        fused = self.fusion(v, a, m)          # (B, 256)

        # --- Temporal (single-step: fake window of 1) ---
        if not window_mode:
            temporal_in = fused.unsqueeze(1)  # (B, 1, 256)
        else:
            temporal_in = fused               # (B, T, 256) already windowed

        temporal_out = self.temporal(temporal_in)  # (B, 256)

        # --- Twin Simulator ---
        next_state = self.twin_sim(temporal_out)   # (B, 256)

        # --- Risk Head ---
        logits = self.risk_head(temporal_out)       # (B, 3)

        return logits, fused, next_state
