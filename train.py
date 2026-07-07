# train.py — Mark 50 edition
# Updated to use Samratth's encoders + temporal + twin simulator
import os
import argparse
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from models.hdt_model import HDTModel
from data.wesad_dataset import get_subject_split_loaders
from utils.vram_guard import vram_check

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WARNING: wandb not installed. Logging to console only.")

# ── Config ──────────────────────────────────────────────────────
CFG = dict(
    lr          = 3e-4,
    batch_size  = 32,        # RTX 5050, 8.5 GB — plenty of room
    epochs      = 30,
    grad_clip   = 1.0,
    ckpt_dir    = "runs/",
    n_classes   = 3,
    twin_loss_w = 0.1,       # weight for twin simulator self-supervised loss
    device      = "cuda" if torch.cuda.is_available() else "cpu",
)

parser = argparse.ArgumentParser()
parser.add_argument("--no_voice",    action="store_true")
parser.add_argument("--no_activity", action="store_true")
parser.add_argument("--no_mobility", action="store_true")
parser.add_argument("--run_name",    default="full_fusion_v2")
parser.add_argument("--use_encoders", action="store_true", default=False,
                    help="Use learned activity/mobility encoders (needs raw features)")
args = parser.parse_args()


import numpy as np

# Load real CREMA-D voice embeddings (pre-projected 256-d)
_voice_np = np.load("data/embeddings/voice_trained.npy")  # (240, 256)
_voice_bank = torch.tensor(_voice_np, dtype=torch.float32)
print(f"Loaded {_voice_bank.shape[0]} voice embeddings ({_voice_bank.shape[1]}-d)")

def get_voice_emb(batch_size, device):
    """Sample random real voice embeddings from CREMA-D bank."""
    idx = torch.randint(0, len(_voice_bank), (batch_size,))
    return _voice_bank[idx].to(device)


def main():
    if WANDB_AVAILABLE:
        wandb.init(project="hdt-fm", name=args.run_name, config=CFG)

    os.makedirs(CFG["ckpt_dir"], exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────
    train_loader, val_loader, scaler = get_subject_split_loaders(
        batch_size=CFG["batch_size"],
        val_subjects=[15, 16, 17]
    )

    # ── Model ────────────────────────────────────────────────────
    model = HDTModel(
        n_classes=CFG["n_classes"],
        use_encoders=args.use_encoders,
    ).to(CFG["device"])

    print(f"Device: {CFG['device']}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    total_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {total_p:,}")

    # ── Optimizer ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG["lr"], weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["epochs"]
    )

    # ── Loss ─────────────────────────────────────────────────────
    class_counts  = torch.tensor([71640.0, 23064.0, 40946.0])
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(CFG["device"])
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    twin_loss_fn = nn.HuberLoss()
    print(f"Class weights: {class_weights.cpu().numpy().round(4)}")

    # ── fp16 ─────────────────────────────────────────────────────
    scaler_amp = torch.amp.GradScaler("cuda", enabled=(CFG["device"] == "cuda"))

    # ── Resume ───────────────────────────────────────────────────
    start_epoch = 0
    best_val_f1 = 0.0
    ckpts = sorted([f for f in os.listdir(CFG["ckpt_dir"])
                    if f.endswith(".pt") and "best" not in f])
    if ckpts:
        path = os.path.join(CFG["ckpt_dir"], ckpts[-1])
        ckpt = torch.load(path, map_location=CFG["device"])
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {ckpts[-1]} (epoch {start_epoch})")

    # ── Train ────────────────────────────────────────────────────
    for epoch in range(start_epoch, CFG["epochs"]):
        model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_twin = 0.0
        n_batches = 0

        for batch_data in train_loader:
            # Handle both dict and tuple formats from dataloader
            if isinstance(batch_data, dict):
                activity = batch_data["activity"].to(CFG["device"])
                mobility = batch_data["mobility"].to(CFG["device"])
                labels   = batch_data["label"].to(CFG["device"])
            else:
                activity, mobility, labels = [x.to(CFG["device"]) for x in batch_data]

            voice_emb = get_voice_emb(activity.shape[0], CFG["device"])

            if args.no_voice:    voice_emb = torch.zeros_like(voice_emb)
            if args.no_activity: activity = torch.zeros_like(activity)
            if args.no_mobility: mobility = torch.zeros_like(mobility)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(CFG["device"] == "cuda")):
                logits, fused, next_state, recon = model(voice_emb=voice_emb, activity=activity, mobility=mobility)
                loss_ce = ce_loss(logits, labels)
                # Twin self-supervised: reconstruct original activity features
                # activity is (B,128) padded — recon targets first 20 cols (real HRV)
                recon_target = activity[:, :20]  # (B, 20) raw HRV features
                loss_twin = twin_loss_fn(recon, recon_target)
                loss = loss_ce + CFG["twin_loss_w"] * loss_twin

            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            scaler_amp.step(optimizer)
            scaler_amp.update()

            total_loss += loss.item()
            total_ce   += loss_ce.item()
            total_twin += loss_twin.item()
            n_batches  += 1

        vram_check(threshold=0.88)

        # ── Validation ───────────────────────────────────────────
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_data in val_loader:
                if isinstance(batch_data, dict):
                    activity = batch_data["activity"].to(CFG["device"])
                    mobility = batch_data["mobility"].to(CFG["device"])
                    labels   = batch_data["label"].to(CFG["device"])
                else:
                    activity, mobility, labels = [x.to(CFG["device"]) for x in batch_data]

                voice_emb = get_voice_emb(activity.shape[0], CFG["device"])
                if args.no_voice:    voice_emb = torch.zeros_like(voice_emb)
                if args.no_activity: activity = torch.zeros_like(activity)
                if args.no_mobility: mobility = torch.zeros_like(mobility)

                with torch.amp.autocast("cuda", enabled=(CFG["device"] == "cuda")):
                    logits, _, _, _ = model(voice_emb=voice_emb, activity=activity, mobility=mobility)
                all_preds.extend(logits.argmax(-1).cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        val_f1   = f1_score(all_labels, all_preds, average="weighted")
        avg_loss = total_loss / n_batches
        avg_ce   = total_ce / n_batches
        avg_twin = total_twin / n_batches

        print(f"Epoch {epoch:03d} | loss={avg_loss:.4f} "
              f"(CE={avg_ce:.4f} twin={avg_twin:.4f}) | val_f1={val_f1:.4f}")

        if WANDB_AVAILABLE:
            wandb.log({
                "loss": avg_loss, "ce_loss": avg_ce, "twin_loss": avg_twin,
                "val_f1": val_f1, "epoch": epoch,
            })

        scheduler.step()

        # ── Checkpoint ───────────────────────────────────────────
        torch.save({
            "epoch": epoch, "model": model.state_dict(),
            "optim": optimizer.state_dict(), "val_f1": val_f1,
        }, os.path.join(CFG["ckpt_dir"], f"ckpt_epoch{epoch:03d}.pt"))

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(),
                       os.path.join(CFG["ckpt_dir"], "best_model.pt"))
            print(f"  ★ New best model (val_f1={val_f1:.4f})")

    print(f"\nTraining complete. Best val F1: {best_val_f1:.4f}")
    if WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()