# models/fusion.py
import torch
import torch.nn as nn

class CrossModalFusion(nn.Module):
    """
    3 modalities → 256-d unified health representation.
    Q = voice, K/V = activity + mobility (concatenated)
    """
    def __init__(self, voice_dim=256, activity_dim=128, mobility_dim=64,
                 out_dim=256, n_heads=4, dropout=0.1):
        super().__init__()
        kv_dim = activity_dim + mobility_dim  # 192

        # project all to same dim for attention
        self.q_proj  = nn.Linear(voice_dim,    out_dim)
        self.kv_proj = nn.Linear(kv_dim,       out_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=out_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)

        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
        )

    def forward(self, voice, activity, mobility):
        # voice:    (B, 256)
        # activity: (B, 128)
        # mobility: (B, 64)
        kv_raw = torch.cat([activity, mobility], dim=-1)  # (B, 192)

        q  = self.q_proj(voice).unsqueeze(1)    # (B, 1, 256)
        kv = self.kv_proj(kv_raw).unsqueeze(1)  # (B, 1, 256)

        attn_out, _ = self.attn(q, kv, kv)      # (B, 1, 256)
        attn_out    = attn_out.squeeze(1)        # (B, 256)

        out = self.norm1(attn_out + self.q_proj(voice))  # residual
        out = self.norm2(out + self.ffn(out))
        return out  # (B, 256)