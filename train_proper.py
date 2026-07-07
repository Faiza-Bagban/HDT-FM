# train_proper.py — subject-held-out clean training
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import collections
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from models.hdt_model import HDTModel
from data.wesad_dataset import get_subject_split_loaders

CFG = dict(
    lr         = 3e-4,
    batch_size = 16,
    epochs     = 30,
    grad_clip  = 1.0,
    ckpt_dir   = "runs_proper",
    n_classes  = 3,
    device     = "cuda" if torch.cuda.is_available() else "cpu",
)

def get_dummy_audio(batch_size, device):
    return torch.randn(batch_size, 16000).to(device)

def main():
    os.makedirs(CFG["ckpt_dir"], exist_ok=True)

    train_loader, val_loader, scaler = get_subject_split_loaders(
        batch_size=CFG["batch_size"],
        val_subjects=[15, 16, 17]
    )

    model = HDTModel(n_classes=CFG["n_classes"]).to(CFG["device"])
    print(f"Device: {CFG['device']}")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG["lr"], weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["epochs"]
    )

    class_counts  = torch.tensor([71640.0, 23064.0, 40946.0])
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(CFG["device"])
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    scaler_amp = torch.amp.GradScaler('cuda', enabled=(CFG["device"] == "cuda"))

    # ── NO auto-resume — fresh start ─────────────────────────────────────────
    best_val_f1 = 0.0

    for epoch in range(CFG["epochs"]):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            activity, mobility, labels = batch
            activity = activity.to(CFG["device"])
            mobility = mobility.to(CFG["device"])
            labels   = labels.to(CFG["device"])
            audio    = get_dummy_audio(activity.shape[0], CFG["device"])

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(CFG["device"] == "cuda")):
                logits, _ = model(audio, activity, mobility)
                loss = ce_loss(logits, labels)

            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            scaler_amp.step(optimizer)
            scaler_amp.update()
            total_loss += loss.item()

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                activity, mobility, labels = batch
                activity = activity.to(CFG["device"])
                mobility = mobility.to(CFG["device"])
                labels   = labels.to(CFG["device"])
                audio    = get_dummy_audio(activity.shape[0], CFG["device"])

                with torch.amp.autocast('cuda', enabled=(CFG["device"] == "cuda")):
                    logits, _ = model(audio, activity, mobility)

                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        pred_dist  = collections.Counter(all_preds)
        label_dist = collections.Counter(all_labels)
        val_f1     = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        avg_loss   = total_loss / len(train_loader)

        print(f"Epoch {epoch:03d} | loss={avg_loss:.4f} | val_f1={val_f1:.4f} | preds={dict(pred_dist)} | labels={dict(label_dist)}")

        scheduler.step()

        # ── Save ──────────────────────────────────────────────────────────────
        torch.save({
            "epoch":  epoch,
            "model":  model.state_dict(),
            "optim":  optimizer.state_dict(),
            "val_f1": val_f1,
        }, os.path.join(CFG["ckpt_dir"], f"ckpt_epoch{epoch:03d}.pt"))

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(),
                       os.path.join(CFG["ckpt_dir"], "best_model.pt"))
            print(f"  ★ New best (val_f1={val_f1:.4f})")

    print(f"\nDone. Best val F1: {best_val_f1:.4f}")

if __name__ == "__main__":
    main()