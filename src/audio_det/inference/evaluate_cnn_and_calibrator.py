#!/usr/bin/env python3
"""
evaluate_cnn_and_calibrator.py

Evaluate CNN-only and CNN+Calibrator performance on a directory with:
  data_dir/
    real/
    fake/

Workflow:
1) Sample files from each class (optional)
2) CNN-only: score each file using segment sampling + aggregation
3) Calibrated: compute meta-features + apply LR calibrator
4) Report accuracy + confusion matrices (and optionally AUC/top-K metrics)

Example:
python src/inference/evaluate_cnn_and_calibrator.py \
  --cnn_model models/cnn/audio_cnn_balanced_best.pt \
  --data_dir data/audio/processed/real_or_fake \
  --calibrator models/calibrators/agg_lr_real_or_fake.joblib \
  --device cpu \
  --sample_per_class 2000 \
  --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import torch
import torchaudio
from sklearn.metrics import confusion_matrix, roc_auc_score


# ----------------------------
# Model definition (must match training)
# ----------------------------
class SmallResNet(torch.nn.Module):
    """Small residual CNN for spoof detection on log-mel spectrograms."""

    def __init__(self, in_ch: int = 1, base: int = 32):
        """Initialise network.

        Args:
            in_ch: Number of input channels (mono spectrogram = 1).
            base: Base channel width.
        """
        super().__init__()

        def block(cin: int, cout: int, stride: int = 1) -> torch.nn.Module:
            """A simple residual block."""
            return torch.nn.Sequential(
                torch.nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
                torch.nn.BatchNorm2d(cout),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
                torch.nn.BatchNorm2d(cout),
            )

        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, base, 3, 1, 1, bias=False),
            torch.nn.BatchNorm2d(base),
            torch.nn.ReLU(inplace=True),
        )
        self.b1 = block(base, base, 1)
        self.p1 = torch.nn.MaxPool2d(2)

        self.b2 = block(base, base * 2, 1)
        self.down2 = torch.nn.Conv2d(base, base * 2, 1, bias=False)
        self.p2 = torch.nn.MaxPool2d(2)

        self.b3 = block(base * 2, base * 4, 1)
        self.down3 = torch.nn.Conv2d(base * 2, base * 4, 1, bias=False)

        self.relu = torch.nn.ReLU(inplace=True)
        self.gap = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.head = torch.nn.Linear(base * 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape [B, 1, n_mels, time].

        Returns:
            Logits tensor of shape [B].
        """
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
        return self.head(x).squeeze(-1)


def load_cnn_model(model_path: str, device: torch.device) -> Tuple[torch.nn.Module, Dict]:
    """Load CNN checkpoint and config.

    Args:
        model_path: Path to CNN .pt checkpoint.
        device: Device to place model on.

    Returns:
        Tuple of (model, config dict).
    """
    ckpt = torch.load(model_path, map_location=device)
    cfg = ckpt["config"]
    model = SmallResNet(in_ch=1, base=32).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


def build_transforms(cfg: Dict, device: torch.device):
    """Build mel-spectrogram and amplitude-to-dB transforms.

    Args:
        cfg: CNN config dictionary.
        device: Torch device.

    Returns:
        Tuple of (melspec, amp_to_db).
    """
    melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=int(cfg["sample_rate"]),
        n_fft=int(cfg["n_fft"]),
        hop_length=int(cfg["hop_length"]),
        win_length=int(cfg["win_length"]),
        f_min=float(cfg["f_min"]),
        f_max=float(cfg["f_max"]),
        n_mels=int(cfg["n_mels"]),
        power=2.0,
        center=True,
        pad_mode="reflect",
    ).to(device)

    amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80).to(device)
    return melspec, amp_to_db


def list_audio_files(
    class_dir: Path,
    exts: Sequence[str],
    recursive: bool = True,
) -> List[Path]:
    """List audio files under a class directory.

    Args:
        class_dir: Directory containing audio files (or nested dirs).
        exts: Allowed file extensions.
        recursive: Whether to search recursively.

    Returns:
        Sorted list of audio file paths.
    """
    exts_set = {e.lower() for e in exts}
    pattern = "**/*" if recursive else "*"
    files = [p for p in class_dir.glob(pattern) if p.is_file() and p.suffix.lower() in exts_set]
    return sorted(files)


def to_mono(wav: torch.Tensor) -> torch.Tensor:
    """Convert waveform to mono [1, T].

    Args:
        wav: Waveform tensor [T] or [C, T].

    Returns:
        Mono waveform tensor [1, T].
    """
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav


def resample_if_needed(wav: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    """Resample waveform if needed.

    Args:
        wav: Waveform [1, T].
        sr: Original sample rate.
        target_sr: Target sample rate.

    Returns:
        Resampled waveform [1, T].
    """
    if sr == target_sr:
        return wav
    return torchaudio.transforms.Resample(sr, target_sr)(wav)


def center_crop_or_pad(wav: torch.Tensor, target_len: int) -> torch.Tensor:
    """Center crop or pad waveform to target length.

    Args:
        wav: Waveform [1, T].
        target_len: Target number of samples.

    Returns:
        Waveform [1, target_len].
    """
    t = wav.size(-1)
    if t > target_len:
        start = (t - target_len) // 2
        return wav[:, start : start + target_len]
    if t < target_len:
        return torch.nn.functional.pad(wav, (0, target_len - t), mode="constant", value=0.0)
    return wav


def segment_start_positions(total_len: int, clip_len: int, n_segments: int) -> List[int]:
    """Compute evenly spaced segment start positions.

    Args:
        total_len: Total waveform length in samples.
        clip_len: Clip length in samples.
        n_segments: Number of segments to sample.

    Returns:
        List of start indices.
    """
    if n_segments <= 1:
        return [max(0, (total_len - clip_len) // 2)]
    if total_len <= clip_len:
        return [0] * n_segments
    max_start = total_len - clip_len
    return [int(round(i * max_start / (n_segments - 1))) for i in range(n_segments)]


def compute_silence_ratio(wav: torch.Tensor) -> float:
    """Compute a simple silence ratio heuristic.

    Args:
        wav: Waveform [1, T].

    Returns:
        Fraction of samples under a dynamic amplitude threshold.
    """
    x = wav.abs().squeeze(0)
    thr = max(0.005, float(x.std().item()) * 0.15)
    return float((x < thr).float().mean().item())


@torch.no_grad()
def score_segments_cnn(
    model: torch.nn.Module,
    cfg: Dict,
    melspec,
    amp_to_db,
    wav: torch.Tensor,
    sr: int,
    clip_seconds: float,
    n_segments: int,
    device: torch.device,
) -> np.ndarray:
    """Score multiple segments of a file with the CNN.

    Args:
        model: CNN model.
        cfg: CNN config.
        melspec: MelSpectrogram transform.
        amp_to_db: AmplitudeToDB transform.
        wav: Waveform [1, T] on CPU or device.
        sr: Sample rate of wav.
        clip_seconds: Clip length in seconds.
        n_segments: Number of segments to score.
        device: Torch device for inference.

    Returns:
        Numpy array of segment probabilities shape [n_segments].
    """
    target_sr = int(cfg["sample_rate"])
    wav = to_mono(wav)
    wav = resample_if_needed(wav, sr, target_sr)
    wav = wav.to(device)

    clip_len = int(target_sr * clip_seconds)
    total_len = wav.size(-1)
    starts = segment_start_positions(total_len, clip_len, n_segments)

    probs: List[float] = []
    for s in starts:
        clip = wav[:, s : s + clip_len]
        clip = center_crop_or_pad(clip, clip_len)

        mel = melspec(clip)
        mel_db = amp_to_db(mel)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

        x = mel_db.unsqueeze(0)  # [1, 1, n_mels, time]
        logits = model(x)
        probs.append(float(torch.sigmoid(logits).item()))

    return np.asarray(probs, dtype=np.float32)


def aggregate_cnn_probs(seg_probs: np.ndarray, agg: str) -> float:
    """Aggregate segment probabilities into a single file score.

    Args:
        seg_probs: Array of segment probabilities.
        agg: Aggregation method: "median", "mean", "max".

    Returns:
        Aggregated probability.
    """
    if seg_probs.size == 0:
        return float("nan")
    agg = agg.lower()
    if agg == "median":
        return float(np.median(seg_probs))
    if agg == "mean":
        return float(np.mean(seg_probs))
    if agg == "max":
        return float(np.max(seg_probs))
    raise ValueError("agg must be one of: median, mean, max")


def compute_calibrator_features(
    seg_probs: np.ndarray,
    total_seconds: float,
    silence_ratio: float,
) -> np.ndarray:
    """Compute calibrator meta-features from CNN segment probabilities.

    Args:
        seg_probs: Segment probabilities array.
        total_seconds: File duration in seconds.
        silence_ratio: Silence fraction heuristic.

    Returns:
        Feature vector shape [5] in order:
            [cnn_median, cnn_max, cnn_var, total_seconds, silence_ratio]
    """
    return np.asarray(
        [
            float(np.median(seg_probs)),
            float(np.max(seg_probs)),
            float(np.var(seg_probs)),
            float(total_seconds),
            float(silence_ratio),
        ],
        dtype=np.float32,
    )


def compute_metrics_at_threshold(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> Dict:
    """Compute accuracy/precision and confusion matrix at a threshold.

    Args:
        y_true: True labels (0=real, 1=fake).
        probs: Predicted probabilities for fake class.
        threshold: Decision threshold.

    Returns:
        Metrics dict including confusion matrix.
    """
    preds = (probs >= threshold).astype(int)

    real_mask = y_true == 0
    fake_mask = y_true == 1

    real_acc = float((preds[real_mask] == 0).mean()) if real_mask.any() else float("nan")
    fake_acc = float((preds[fake_mask] == 1).mean()) if fake_mask.any() else float("nan")
    overall_acc = float((preds == y_true).mean()) if y_true.size else float("nan")

    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0

    cm = confusion_matrix(y_true, preds, labels=[0, 1]).tolist()
    return {
        "threshold": float(threshold),
        "real_accuracy": real_acc,
        "fake_accuracy": fake_acc,
        "overall_accuracy": overall_acc,
        "overall_precision": precision,
        "confusion_matrix": cm,
        "confusion_matrix_labels": ["real", "fake"],
    }


def compute_topk_metrics(y_true: np.ndarray, probs: np.ndarray, review_frac: float) -> Dict:
    """Compute top-K review metrics (rank-based).

    Args:
        y_true: True labels (0=real, 1=fake).
        probs: Scores/probabilities (higher => more suspicious).
        review_frac: Fraction to review (e.g., 0.1 = top 10%).

    Returns:
        Dict containing reviewed_fake_rate and fake_catch_rate among reviewed.
    """
    n = len(y_true)
    k = max(1, int(round(n * review_frac)))
    idx = np.argsort(-probs)[:k]
    reviewed = y_true[idx]

    fake_total = int((y_true == 1).sum())
    fake_caught = int((reviewed == 1).sum())
    fake_catch_rate = float(fake_caught / fake_total) if fake_total else float("nan")
    reviewed_fake_rate = float((reviewed == 1).mean()) if reviewed.size else float("nan")

    return {
        "review_frac": float(review_frac),
        "n_total": int(n),
        "n_review": int(k),
        "fake_total": int(fake_total),
        "fake_caught": int(fake_caught),
        "fake_catch_rate": fake_catch_rate,
        "reviewed_fake_rate": reviewed_fake_rate,
    }


def sample_files_per_class(
    real_files: List[Path],
    fake_files: List[Path],
    sample_per_class: Optional[int],
    seed: Optional[int],
) -> Tuple[List[Path], List[Path]]:
    """Sample an equal number of files from each class.

    Args:
        real_files: List of real file paths.
        fake_files: List of fake file paths.
        sample_per_class: Number to sample per class (None => use all).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (sampled_real_files, sampled_fake_files).
    """
    if sample_per_class is None:
        return real_files, fake_files

    rng = random.Random(seed)
    real = real_files
    fake = fake_files

    if sample_per_class < len(real):
        real = rng.sample(real, sample_per_class)
    if sample_per_class < len(fake):
        fake = rng.sample(fake, sample_per_class)

    return sorted(real), sorted(fake)


def load_calibrator(calibrator_path: str):
    """Load a joblib calibrator, supporting either raw estimator or a dict bundle.

    Args:
        calibrator_path: Path to .joblib.

    Returns:
        An object with predict_proba(X) -> [n,2].
    """
    obj = joblib.load(calibrator_path)
    if isinstance(obj, dict) and "aggregator" in obj:
        return obj["aggregator"]
    return obj


def parse_thresholds(thresholds_str: str) -> List[float]:
    """Parse comma-separated thresholds.

    Args:
        thresholds_str: Comma-separated threshold string.

    Returns:
        List of float thresholds.
    """
    out: List[float] = []
    for t in thresholds_str.split(","):
        t = t.strip()
        if not t:
            continue
        out.append(float(t))
    if not out:
        out = [0.5]
    return out


def main() -> None:
    """Run evaluation and print JSON report."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn_model", required=True, help="Path to CNN checkpoint .pt")
    ap.add_argument("--data_dir", required=True, help="Directory containing real/ and fake/")
    ap.add_argument("--calibrator", default=None, help="Optional path to LR calibrator .joblib")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--exts", default=".flac,.wav", help="Comma-separated extensions (e.g. .flac,.wav,.m4a)")
    ap.add_argument("--recursive", action="store_true", help="Search recursively under real/ and fake/")
    ap.add_argument("--sample_per_class", type=int, default=None, help="Sample N files per class")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clip_seconds", type=float, default=None, help="Override clip_seconds (else use cfg)")
    ap.add_argument("--n_segments", type=int, default=6)
    ap.add_argument("--cnn_agg", default="median", choices=["median", "mean", "max"])
    ap.add_argument("--thresholds", default="0.3,0.5,0.7,0.85")
    ap.add_argument("--report_out", default=None, help="Optional path to write JSON report")
    args = ap.parse_args()

    device = torch.device(args.device)

    model, cfg = load_cnn_model(args.cnn_model, device=device)
    melspec, amp_to_db = build_transforms(cfg, device=device)

    clip_seconds = float(args.clip_seconds) if args.clip_seconds is not None else float(cfg.get("clip_seconds", 4.0))
    thresholds = parse_thresholds(args.thresholds)
    exts = tuple(e.strip().lower() for e in args.exts.split(",") if e.strip())

    data_dir = Path(args.data_dir)
    real_dir = data_dir / "real"
    fake_dir = data_dir / "fake"
    if not real_dir.is_dir() or not fake_dir.is_dir():
        raise ValueError(f"Expected {data_dir}/real and {data_dir}/fake to exist.")

    real_files = list_audio_files(real_dir, exts=exts, recursive=args.recursive)
    fake_files = list_audio_files(fake_dir, exts=exts, recursive=args.recursive)

    real_files, fake_files = sample_files_per_class(real_files, fake_files, args.sample_per_class, args.seed)

    files: List[Tuple[Path, int]] = [(p, 0) for p in real_files] + [(p, 1) for p in fake_files]
    if not files:
        raise RuntimeError("No audio files found.")

    # Score files
    y_true: List[int] = []
    cnn_probs: List[float] = []
    cal_probs: List[float] = []

    calibrator = load_calibrator(args.calibrator) if args.calibrator else None
    X_feats: List[np.ndarray] = []

    for p, lab in files:
        wav, sr = torchaudio.load(str(p))
        wav = to_mono(wav)

        # Duration/silence computed on resampled waveform for consistency
        target_sr = int(cfg["sample_rate"])
        wav_rs = resample_if_needed(wav, sr, target_sr)
        total_seconds = float(wav_rs.size(-1) / target_sr)
        silence_ratio = compute_silence_ratio(wav_rs)

        seg_probs = score_segments_cnn(
            model=model,
            cfg=cfg,
            melspec=melspec,
            amp_to_db=amp_to_db,
            wav=wav,
            sr=sr,
            clip_seconds=clip_seconds,
            n_segments=int(args.n_segments),
            device=device,
        )

        file_prob = aggregate_cnn_probs(seg_probs, args.cnn_agg)

        y_true.append(lab)
        cnn_probs.append(file_prob)

        if calibrator is not None:
            feats = compute_calibrator_features(seg_probs, total_seconds, silence_ratio)
            X_feats.append(feats)

    y_true_np = np.asarray(y_true, dtype=np.int64)
    cnn_probs_np = np.asarray(cnn_probs, dtype=np.float32)

    # CNN metrics
    cnn_auc = float(roc_auc_score(y_true_np, cnn_probs_np)) if len(np.unique(y_true_np)) > 1 else float("nan")
    cnn_threshold_metrics = [compute_metrics_at_threshold(y_true_np, cnn_probs_np, t) for t in thresholds]
    cnn_topk = [compute_topk_metrics(y_true_np, cnn_probs_np, f) for f in (0.05, 0.10, 0.20)]

    report: Dict = {
        "data_dir": str(data_dir),
        "n_total": int(len(y_true_np)),
        "n_real": int((y_true_np == 0).sum()),
        "n_fake": int((y_true_np == 1).sum()),
        "clip_seconds": float(clip_seconds),
        "n_segments": int(args.n_segments),
        "cnn_agg": args.cnn_agg,
        "thresholds": thresholds,
        "cnn_only": {
            "auc": cnn_auc,
            "threshold_metrics": cnn_threshold_metrics,
            "topk_metrics": cnn_topk,
        },
    }

    # Calibrator metrics (optional)
    if calibrator is not None:
        X = np.stack(X_feats, axis=0) if X_feats else np.zeros((0, 5), dtype=np.float32)
        cal_probs_np = calibrator.predict_proba(X)[:, 1].astype(np.float32)

        cal_auc = float(roc_auc_score(y_true_np, cal_probs_np)) if len(np.unique(y_true_np)) > 1 else float("nan")
        cal_threshold_metrics = [compute_metrics_at_threshold(y_true_np, cal_probs_np, t) for t in thresholds]
        cal_topk = [compute_topk_metrics(y_true_np, cal_probs_np, f) for f in (0.05, 0.10, 0.20)]

        report["calibrated"] = {
            "calibrator_path": str(args.calibrator),
            "feature_names": ["cnn_median", "cnn_max", "cnn_var", "total_seconds", "silence_ratio"],
            "auc": cal_auc,
            "threshold_metrics": cal_threshold_metrics,
            "topk_metrics": cal_topk,
        }

    # Print + optionally save
    print(json.dumps(report, indent=2))

    if args.report_out:
        out_path = Path(args.report_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
