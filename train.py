# train.py
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score

from models.hdt_model import HDTModel
from data.wesad_dataset import WESADDataset
from utils.vram_guard import vram_check

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WARNING: wandb not installed. Run: pip install wandb. Logging to console only.")

# ── Config ────────────────────────────────────────────────────────────────────
CFG = dict(
    lr          = 3e-4,
    batch_size  = 16,        # safe on RTX 4050 6.4 GB
    epochs      = 30,
    grad_clip   = 1.0,
    ckpt_dir = "runs_proper/",
    n_classes   = 3,
    device      = "cuda" if torch.cuda.is_available() else "cpu",
)

# ── Arg parsing (for ablation variants) ──────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--no_voice",    action="store_true", help="Zero out voice modality")
parser.add_argument("--no_activity", action="store_true", help="Zero out activity modality")
parser.add_argument("--no_mobility", action="store_true", help="Zero out mobility modality")
parser.add_argument("--run_name",    default="full_fusion", help="WandB run name")
args = parser.parse_args()


def get_dummy_audio(batch_size, device):
    """Placeholder 1-sec audio @ 16kHz. Replace with real loader when RAVDESS aligned."""
    return torch.randn(batch_size, 16000).to(device)


def main():
    # ── WandB init ────────────────────────────────────────────────────────────
    if WANDB_AVAILABLE:
        wandb.init(project="hdt-fm", name=args.run_name, config=CFG)

    os.makedirs(CFG["ckpt_dir"], exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────────────────────
    from data.wesad_dataset import get_subject_split_loaders
    train_loader, val_loader, scaler = get_subject_split_loaders(
    batch_size=CFG["batch_size"],
    val_subjects=[15, 16, 17]
    )
    # ── Model ─────────────────────────────────────────────────────────────────
    model = HDTModel(n_classes=CFG["n_classes"]).to(CFG["device"])
    print(f"Device: {CFG['device']}")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG["lr"], weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["epochs"]
    )

    # ── Loss: weighted CE for class imbalance ─────────────────────────────────
    # counts: baseline=71640, amusement=23064, stress=40946
    class_counts  = torch.tensor([71640.0, 23064.0, 40946.0])
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(CFG["device"])
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    print(f"Class weights: {class_weights.cpu().numpy().round(4)}")

    # ── fp16 scaler ───────────────────────────────────────────────────────────
    scaler = torch.cuda.amp.GradScaler(enabled=(CFG["device"] == "cuda"))

    # ── Auto-resume from latest checkpoint ────────────────────────────────────
    start_epoch = 0
    ckpts = sorted([f for f in os.listdir(CFG["ckpt_dir"]) if f.endswith(".pt") and "best" not in f])
    if ckpts:
        ckpt_path = os.path.join(CFG["ckpt_dir"], ckpts[-1])
        ckpt = torch.load(ckpt_path, map_location=CFG["device"])
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from checkpoint: {ckpts[-1]} (epoch {start_epoch})")

    best_val_f1 = 0.0

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, CFG["epochs"]):
        model.train()
        total_loss   = 0.0
        total_batches = 0

        for batch in train_loader:
            activity = batch["activity"].to(CFG["device"])
            mobility = batch["mobility"].to(CFG["device"])
            labels   = batch["label"].to(CFG["device"])
            audio    = get_dummy_audio(activity.shape[0], CFG["device"])

            # ablation: zero out disabled modalities
            if args.no_voice:    audio    = torch.zeros_like(audio)
            if args.no_activity: activity = torch.zeros_like(activity)
            if args.no_mobility: mobility = torch.zeros_like(mobility)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(CFG["device"] == "cuda")):
                logits, _ = model(audio, activity, mobility)
                loss = ce_loss(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

            total_loss    += loss.item()
            total_batches += 1

        # VRAM guard after each epoch
        vram_check(threshold=0.88)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                activity = batch["activity"].to(CFG["device"])
                mobility = batch["mobility"].to(CFG["device"])
                labels   = batch["label"].to(CFG["device"])
                audio    = get_dummy_audio(activity.shape[0], CFG["device"])

                if args.no_voice:    audio    = torch.zeros_like(audio)
                if args.no_activity: activity = torch.zeros_like(activity)
                if args.no_mobility: mobility = torch.zeros_like(mobility)

                with torch.cuda.amp.autocast(enabled=(CFG["device"] == "cuda")):
                    logits, _ = model(audio, activity, mobility)

                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        val_f1   = f1_score(all_labels, all_preds, average="weighted")
        avg_loss = total_loss / total_batches

        print(f"Epoch {epoch:03d} | loss={avg_loss:.4f} | val_f1={val_f1:.4f}")

        if WANDB_AVAILABLE:
            wandb.log({"loss": avg_loss, "val_f1": val_f1, "epoch": epoch})

        scheduler.step()

        # ── Save checkpoint every epoch ───────────────────────────────────────
        torch.save({
            "epoch":   epoch,
            "model":   model.state_dict(),
            "optim":   optimizer.state_dict(),
            "val_f1":  val_f1,
        }, os.path.join(CFG["ckpt_dir"], f"ckpt_epoch{epoch:03d}.pt"))

        # ── Save best model separately ────────────────────────────────────────
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                model.state_dict(),
                os.path.join(CFG["ckpt_dir"], "best_model.pt")
            )
            print(f"  ★ New best model saved (val_f1={val_f1:.4f})")

    print(f"\nTraining complete. Best val F1: {best_val_f1:.4f}")

    if WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()