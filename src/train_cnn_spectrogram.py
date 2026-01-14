#!/usr/bin/env python3
"""
train_cnn_spectrogram.py

Minimal CNN (log-mel spectrogram) training script for audio spoof / authenticity risk.

Data layout (same as your baseline):
  data/
    real/  (bonafide)
    fake/  (spoof)

Outputs:
  models/audio_cnn_mel.pt   (torch state_dict + config)

Python 3.12 compatible if you use torch/torchaudio 2.2+.

Usage:
    python src/train_cnn_spectrogram.py \
    --data_dir data/audio/train \
    --out_path models/audio_cnn_balanced.pt \
    --epochs 30 \
    --batch_size 32 \
    --clip_seconds 4.0 \
    --lr 1e-4 \
    --num_workers 4 \
    --device cuda


Install:
  pip install torch torchaudio numpy tqdm
  # (plus whatever you already have; this script does not require librosa/scikit-learn)
"""

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler

from tqdm import tqdm


# -------------------------
# Config
# -------------------------
@dataclass
class TrainConfig:
    data_dir: str = "data"
    out_path: str = "models/audio_cnn_mel.pt"

    sample_rate: int = 16000
    clip_seconds: float = 4.0  # fixed-length crop; good starting point for spoof artifacts
    n_mels: int = 80
    n_fft: int = 1024
    hop_length: int = 160       # 10ms at 16k
    win_length: int = 400       # 25ms at 16k
    f_min: int = 20
    f_max: int = 7600

    batch_size: int = 32
    epochs: int = 12
    lr: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 42

    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Splits
    val_frac: float = 0.2

    # Training tricks
    pos_weight: float = 1.0  # set >1 if fake is underrepresented
    mixup_alpha: float = 0.0  # 0 disables; can help later (try 0.2)


# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_audio_files(folder: Path) -> List[Path]:
    exts = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
    return [p for p in folder.glob("*") if p.is_file() and p.suffix.lower() in exts]


def train_val_split(paths: List[Path], val_frac: float, seed: int) -> Tuple[List[Path], List[Path]]:
    rng = random.Random(seed)
    paths = paths[:]
    rng.shuffle(paths)
    n_val = max(1, int(len(paths) * val_frac))
    return paths[n_val:], paths[:n_val]


# -------------------------
# Dataset
# -------------------------
class SpoofDataset(Dataset):
    """
    Returns:
      x: log-mel spectrogram tensor [1, n_mels, time]
      y: label float tensor [] where 0=real, 1=fake
    """
    def __init__(
        self,
        real_paths: List[Path],
        fake_paths: List[Path],
        cfg: TrainConfig,
        train: bool,
    ):
        self.cfg = cfg
        self.train = train
        self.items = [(p, 0) for p in real_paths] + [(p, 1) for p in fake_paths]
        random.shuffle(self.items)

        self.resampler_cache = {}  # (orig_sr -> Resample module)

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
            n_mels=cfg.n_mels,
            power=2.0,
            center=True,
            pad_mode="reflect",
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)

        self.target_len = int(cfg.sample_rate * cfg.clip_seconds)

    def __len__(self):
        return len(self.items)

    def _resample_if_needed(self, wav: torch.Tensor, orig_sr: int) -> torch.Tensor:
        if orig_sr == self.cfg.sample_rate:
            return wav
        key = orig_sr
        if key not in self.resampler_cache:
            self.resampler_cache[key] = torchaudio.transforms.Resample(orig_sr, self.cfg.sample_rate)
        return self.resampler_cache[key](wav)

    def _to_mono(self, wav: torch.Tensor) -> torch.Tensor:
        # wav shape: [channels, time]
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if wav.size(0) == 1:
            return wav
        return wav.mean(dim=0, keepdim=True)

    def _crop_or_pad(self, wav: torch.Tensor) -> torch.Tensor:
        # wav shape [1, time]
        t = wav.size(-1)
        if t == self.target_len:
            return wav
        if t > self.target_len:
            if self.train:
                start = random.randint(0, t - self.target_len)
            else:
                start = (t - self.target_len) // 2
            return wav[:, start : start + self.target_len]
        # pad (reflect padding works well, but constant also ok)
        pad = self.target_len - t
        return torch.nn.functional.pad(wav, (0, pad), mode="constant", value=0.0)

    def _augment(self, wav: torch.Tensor) -> torch.Tensor:
        # Keep v1 augmentation light and “insurance-realistic”.
        if not self.train:
            return wav

        # Random gain
        if random.random() < 0.3:
            gain_db = random.uniform(-6.0, 6.0)
            wav = wav * (10.0 ** (gain_db / 20.0))

        # Add small noise
        if random.random() < 0.3:
            noise = torch.randn_like(wav) * random.uniform(0.001, 0.01)
            wav = wav + noise

        # Random lowpass-ish effect (very mild)
        if random.random() < 0.15:
            # crude: downsample+upsample to mimic bandwidth limitation
            # e.g. 8k telephone band
            tmp_sr = random.choice([8000, 12000])
            down = torchaudio.transforms.Resample(self.cfg.sample_rate, tmp_sr)
            up = torchaudio.transforms.Resample(tmp_sr, self.cfg.sample_rate)
            wav = up(down(wav))

        return wav.clamp(-1.0, 1.0)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        wav, sr = torchaudio.load(str(path))
        wav = self._to_mono(wav)
        wav = self._resample_if_needed(wav, sr)
        wav = self._crop_or_pad(wav)
        wav = self._augment(wav)

        # Log-mel
        mel = self.melspec(wav)          # [1, n_mels, time]
        mel_db = self.amp_to_db(mel)     # log scale

        # Normalise per-sample (stable + helps generalisation)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

        y = torch.tensor(float(label), dtype=torch.float32)
        return mel_db, y


# -------------------------
# Model
# -------------------------
class SmallResNet(nn.Module):
    """
    Minimal 2D CNN for log-mel spoof classification.
    Input: [B, 1, n_mels, T]
    Output: logits [B]
    """
    def __init__(self, in_ch: int = 1, base: int = 32):
        super().__init__()

        def block(cin, cout, stride=1):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(cout),
            )

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )

        self.b1 = block(base, base, stride=1)
        self.p1 = nn.MaxPool2d(2)

        self.b2 = block(base, base * 2, stride=1)
        self.down2 = nn.Conv2d(base, base * 2, kernel_size=1, bias=False)
        self.p2 = nn.MaxPool2d(2)

        self.b3 = block(base * 2, base * 4, stride=1)
        self.down3 = nn.Conv2d(base * 2, base * 4, kernel_size=1, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(base * 4, 1)

    def forward(self, x):
        x = self.stem(x)

        r = x
        x = self.b1(x)
        x = self.relu(x + r)
        x = self.p1(x)

        r = self.down2(x)
        x = self.b2(x)
        x = self.relu(x + r)
        x = self.p2(x)

        r = self.down3(x)
        x = self.b3(x)
        x = self.relu(x + r)

        x = self.gap(x).squeeze(-1).squeeze(-1)
        logits = self.head(x).squeeze(-1)
        return logits


# -------------------------
# Training
# -------------------------
@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    probs = []
    ys = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        p = torch.sigmoid(logits)
        probs.append(p.detach().cpu())
        ys.append(yb.detach().cpu())
    probs = torch.cat(probs).numpy()
    ys = torch.cat(ys).numpy()

    # Simple metrics (accuracy per class and overall)
    preds = (probs >= 0.5).astype(np.int32)
    ys_i = ys.astype(np.int32)

    real_mask = ys_i == 0
    fake_mask = ys_i == 1

    real_acc = float((preds[real_mask] == 0).mean()) if real_mask.any() else float("nan")
    fake_acc = float((preds[fake_mask] == 1).mean()) if fake_mask.any() else float("nan")
    overall_acc = float((preds == ys_i).mean())

    # Precision (overall, positive class=fake)
    tp = int(((preds == 1) & (ys_i == 1)).sum())
    fp = int(((preds == 1) & (ys_i == 0)).sum())
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    return {
        "real_accuracy": real_acc,
        "fake_accuracy": fake_acc,
        "overall_accuracy": overall_acc,
        "overall_precision": precision,
        "real_n": int(real_mask.sum()),
        "fake_n": int(fake_mask.sum()),
    }


def mixup(x, y, alpha: float):
    if alpha <= 0:
        return x, y
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x2 = x[idx]
    y2 = y[idx]
    xm = lam * x + (1 - lam) * x2
    ym = lam * y + (1 - lam) * y2
    return xm, ym


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_path", default="models/audio_cnn_mel.pt")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--clip_seconds", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--mixup_alpha", type=float, default=0.0)
    args = ap.parse_args()

    cfg = TrainConfig(
        data_dir=args.data_dir,
        out_path=args.out_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        clip_seconds=args.clip_seconds,
        lr=args.lr,
        seed=args.seed,
        num_workers=args.num_workers,
        mixup_alpha=args.mixup_alpha,
    )

    set_seed(cfg.seed)

    data_dir = Path(cfg.data_dir)
    real_paths = list_audio_files(data_dir / "real")
    fake_paths = list_audio_files(data_dir / "fake")

    if len(real_paths) == 0 or len(fake_paths) == 0:
        raise RuntimeError(
            f"Expected audio in {data_dir/'real'} and {data_dir/'fake'}; "
            f"found real={len(real_paths)} fake={len(fake_paths)}"
        )

    real_tr, real_va = train_val_split(real_paths, cfg.val_frac, cfg.seed)
    fake_tr, fake_va = train_val_split(fake_paths, cfg.val_frac, cfg.seed)

    # If imbalance, you can set pos_weight = (n_real / n_fake) or similar.
    # Here we compute a mild default based on train split.
    n_real = len(real_tr)
    n_fake = len(fake_tr)
    # cfg.pos_weight = float(n_real / max(1, n_fake))
    cfg.pos_weight = 1

    train_ds = SpoofDataset(real_tr, fake_tr, cfg, train=True)
    val_ds = SpoofDataset(real_va, fake_va, cfg, train=False)

    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"  Real train: {len(real_tr)} | Real val: {len(real_va)}")
    print(f"  Fake train: {len(fake_tr)} | Fake val: {len(fake_va)}")
    print(f"  Epochs: {cfg.epochs} | Batch size: {cfg.batch_size}")
    print(f"  Learning rate: {cfg.lr}")

    # --- Balanced sampling to handle class imbalance (approx 50/50 per batch) ---

    # Assuming train_ds returns (x, y) where y is 0=real, 1=fake
    # If your dataset stores labels differently, adjust the label extraction below.

    # 1) Extract labels for every item in the training dataset
    train_labels = []
    for i in range(len(train_ds)):
        _, y = train_ds[i]
        # y might be a tensor; convert safely
        y_int = int(y) if not torch.is_tensor(y) else int(y.item())
        train_labels.append(y_int)

    train_labels = np.array(train_labels, dtype=np.int64)

    # 2) Compute per-sample weights: inverse frequency
    class_counts = np.bincount(train_labels, minlength=2)  # [n_real, n_fake]
    class_weights = 1.0 / np.maximum(class_counts, 1)      # inverse counts

    sample_weights = class_weights[train_labels]
    sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)

    # 3) Sampler draws indices so batches are class-balanced over time
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_ds),   # one epoch ~ same number of samples as dataset
        replacement=True             # allows oversampling minority class
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=sampler,             # <- sampler replaces shuffle
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available()
    )

    device = torch.device(cfg.device)
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
    else:
        print(f"  Running on CPU (training will be slower)")
    print(f"{'='*60}\n")
    
    model = SmallResNet(in_ch=1, base=32).to(device)

    # Weighted BCE for class imbalance (fake = positive class)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg.pos_weight, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_val = -1.0
    out_path = Path(cfg.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.epochs}:")
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"  Training", leave=True)

        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)

            xb, yb = mixup(xb, yb, cfg.mixup_alpha)

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            running += loss.item()
            pbar.set_postfix(loss=running / (pbar.n + 1))

        scheduler.step()

        print("  Evaluating...")
        metrics = evaluate(model, val_loader, device)
        print(f"  Val Accuracy: {metrics['overall_accuracy']:.4f} | Real: {metrics['real_accuracy']:.4f} | Fake: {metrics['fake_accuracy']:.4f}")

        # Track best by val_accuracy (since that's what you're currently missing)
        score = metrics["overall_accuracy"]
        if score > best_val:
            best_val = score
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": asdict(cfg),
                    "val_metrics": metrics,
                },
                out_path
            )
            print(f"  ✓ New best! Saved to {out_path}")

    print(f"\n{'='*60}")
    print(f"Training complete! Best fake accuracy: {best_val:.4f}")
    print(f"Model saved to: {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
