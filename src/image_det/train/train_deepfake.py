#!/usr/bin/env python
"""
Binary image classifier (real vs deepfake) training script using PyTorch.

Folder layout expected:
  data_dir/
    real/
      img1.jpg
      ...
    deepfake/
      img2.jpg
      ...

Example:
  python train_deepfake.py --data_dir S:/dev/Deepfake-vs-Real-v2 --epochs 10 --batch_size 32

    python src/image_det/train/train_deepfake.py \
    --data_dir "data/image_det/CASIA2_organized/train" \
    --out_dir "models/image_det/runs_casia" \
    --epochs 25 \
    --batch_size 32 \
    --model_name efficientnet_b0 \
    --patience 5

"""

import os
import math
import time
import random
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

logger.info("Logging successfully configured.")


# -----------------------------
# Config
# -----------------------------
@dataclass
class TrainConfig:
    data_dir: str
    out_dir: str = "./runs_deepfake"
    epochs: int = 10
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    img_size: int = 224
    val_split: float = 0.15
    seed: int = 42
    model_name: str = "efficientnet_b0"  # "resnet50" also supported
    use_amp: bool = True
    patience: int = 3  # early stopping on val loss
    use_weighted_sampler: bool = False  # Use WeightedRandomSampler for class imbalance
    auto_class_weights: bool = True  # Automatically calculate class weights for loss


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    # logits shape [B, 1] or [B]
    probs = torch.sigmoid(logits.squeeze(1))
    preds = (probs >= 0.5).long()
    return (preds == targets).float().mean().item()


def make_out_dir(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, stamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def stratified_split_indices(targets, val_split: float, seed: int) -> Tuple[list, list]:
    """
    Stratified split for binary labels in ImageFolder.
    """
    idx0 = [i for i, y in enumerate(targets) if y == 0]
    idx1 = [i for i, y in enumerate(targets) if y == 1]

    rng = random.Random(seed)
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    n0_val = int(len(idx0) * val_split)
    n1_val = int(len(idx1) * val_split)

    val_idx = idx0[:n0_val] + idx1[:n1_val]
    train_idx = idx0[n0_val:] + idx1[n1_val:]

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


# -----------------------------
# Model
# -----------------------------
def build_model(model_name: str) -> nn.Module:
    if model_name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        # Replace classifier head
        in_features = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_features, 1)
        return m

    if model_name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, 1)
        return m

    raise ValueError(f"Unsupported model_name={model_name}. Use efficientnet_b0 or resnet50.")


# -----------------------------
# Train / Eval loops
# -----------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).long()

        logits = model(images)
        loss = criterion(logits.squeeze(1), targets.float())

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy_from_logits(logits, targets) * bs
        n += bs

    return {"loss": total_loss / max(n, 1), "acc": total_acc / max(n, 1)}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits.squeeze(1), targets.float())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits.squeeze(1), targets.float())
            loss.backward()
            optimizer.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy_from_logits(logits.detach(), targets) * bs
        n += bs

    return {"loss": total_loss / max(n, 1), "acc": total_acc / max(n, 1)}


# -----------------------------
# Main
# -----------------------------
def main(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = make_out_dir(cfg.out_dir)
    print(f"Device: {device}")
    print(f"Run dir: {run_dir}")
    print(f"Data dir: {cfg.data_dir}")

    # Transforms: strong-ish aug for generalisation
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(cfg.img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.5),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize(int(cfg.img_size * 1.15)),
        transforms.CenterCrop(cfg.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    # IMPORTANT: ImageFolder assigns class indices alphabetically by folder name.
    # With folders: deepfake/ and real/, classes will be ['deepfake', 'real'] -> labels 0 and 1 respectively.
    base_ds = datasets.ImageFolder(cfg.data_dir)

    # Stratified split so class balance stays stable
    targets = [y for _, y in base_ds.samples]
    train_idx, val_idx = stratified_split_indices(targets, cfg.val_split, cfg.seed)

    # Build two datasets with different transforms
    train_ds = datasets.ImageFolder(cfg.data_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(cfg.data_dir, transform=val_tfms)

    train_ds = Subset(train_ds, train_idx)
    val_ds = Subset(val_ds, val_idx)

    print(f"Classes: {base_ds.classes} (label mapping)")
    print(f"Train size: {len(train_ds)} | Val size: {len(val_ds)}")

    # Class weights for imbalance handling
    y_train = [targets[i] for i in train_idx]
    n_pos = sum(1 for y in y_train if y == 1)  # fake (label=1)
    n_neg = sum(1 for y in y_train if y == 0)  # real (label=0)
    
    print(f"\nClass distribution in training:")
    print(f"  Label 0 ({base_ds.classes[0]}): {n_neg} samples")
    print(f"  Label 1 ({base_ds.classes[1]}): {n_pos} samples")
    print(f"  Imbalance ratio: {max(n_neg, n_pos) / min(n_neg, n_pos):.2f}:1")
    
    # Calculate class weights for loss function
    if cfg.auto_class_weights:
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
        print(f"  Using class weight (pos_weight): {pos_weight.item():.3f}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
        print(f"  Using unweighted loss")

    model = build_model(cfg.model_name).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Cosine schedule over total steps (simple + effective)
    steps_per_epoch = math.ceil(len(train_ds) / cfg.batch_size)
    total_steps = max(cfg.epochs * steps_per_epoch, 1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))

    # Create data loaders with optional weighted sampling
    if cfg.use_weighted_sampler:
        # Calculate sample weights (inverse of class frequency)
        class_counts = [n_neg, n_pos]
        sample_weights = [1.0 / class_counts[y] for y in y_train]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        print(f"  Using WeightedRandomSampler for balanced batch sampling")
        
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    best_val_loss = float("inf")
    best_path = os.path.join(run_dir, "best.pt")
    last_path = os.path.join(run_dir, "last.pt")
    bad_epochs = 0

    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=(cfg.use_amp and device.type == "cuda"),
        )

        # Step scheduler per batch-equivalent (simple approximation: step once per epoch * steps_per_epoch)
        # Better: step per batch; but to keep code simpler, we'll step per batch via loop below:
        # We'll do it properly:
        # (We can't easily step inside train_one_epoch without passing scheduler, so we step here in a loop.)
        # Approx: step steps_per_epoch times.
        for _ in range(steps_per_epoch):
            scheduler.step()
            global_step += 1

        val_metrics = evaluate(model, val_loader, criterion, device)

        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"lr={lr_now:.2e} | "
            f"train loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.4f} | "
            f"{dt:.1f}s"
        )

        # Save last
        torch.save(
            {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "cfg": cfg.__dict__},
            last_path,
        )

        # Early stopping & best checkpoint
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            bad_epochs = 0
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "cfg": cfg.__dict__},
                best_path,
            )
            print(f"  -> Saved best to: {best_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"Early stopping: no val loss improvement for {cfg.patience} epochs.")
                break

    print("Done.")
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True, help="Folder containing real/ and deepfake/ subfolders.")
    p.add_argument("--out_dir", type=str, default="./models/image_det/runs_deepfake")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_name", type=str, default="efficientnet_b0", choices=["efficientnet_b0", "resnet50"])
    p.add_argument("--no_amp", action="store_true", help="Disable mixed precision (AMP).")
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--use_weighted_sampler", action="store_true", help="Use WeightedRandomSampler for class imbalance.")
    p.add_argument("--no_auto_class_weights", action="store_true", help="Disable automatic class weights in loss.")

    args = p.parse_args()
    cfg = TrainConfig(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        img_size=args.img_size,
        val_split=args.val_split,
        seed=args.seed,
        model_name=args.model_name,
        use_amp=(not args.no_amp),
        patience=args.patience,
        use_weighted_sampler=args.use_weighted_sampler,
        auto_class_weights=(not args.no_auto_class_weights),
    )
    main(cfg)
