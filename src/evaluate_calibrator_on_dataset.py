#!/usr/bin/env python3
"""
evaluate_calibrator_on_dataset.py

Evaluate a trained logistic regression calibrator on a dataset using
frozen CNN-derived features.

Typical use case:
- Train calibrator on dataset A (e.g. Real-or-Fake)
- Evaluate that calibrator on dataset B (e.g. ASVspoof)
- Measure cross-domain calibration behaviour

Expected directory layout:
  data_dir/
    real/
    fake/

Usage example:
    python src/evaluate_calibrator_on_dataset.py \
  --cnn_model models/cnn/audio_cnn_balanced_best.pt \
  --calibrator models/agg_lr_real_or_fake.joblib \
  --data_dir data/audio/eval \
  --device cuda

"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import torch
import torchaudio
from sklearn.metrics import confusion_matrix, roc_auc_score
from tqdm import tqdm


# ---------------------------------------------------------------------
# CNN architecture (must exactly match training)
# ---------------------------------------------------------------------
class SmallResNet(torch.nn.Module):
    """Small residual CNN used for spoof detection on log-mel spectrograms."""

    def __init__(self, in_ch: int = 1, base: int = 32):
        """
        Initialise the SmallResNet model.

        Args:
            in_ch: Number of input channels (default: 1 for mono spectrogram).
            base: Base channel width for the network.
        """
        super().__init__()

        def block(cin, cout, stride=1):
            """Residual convolutional block."""
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
        self.b1 = block(base, base)
        self.p1 = torch.nn.MaxPool2d(2)
        self.b2 = block(base, base * 2)
        self.down2 = torch.nn.Conv2d(base, base * 2, 1, bias=False)
        self.p2 = torch.nn.MaxPool2d(2)
        self.b3 = block(base * 2, base * 4)
        self.down3 = torch.nn.Conv2d(base * 2, base * 4, 1, bias=False)
        self.relu = torch.nn.ReLU(inplace=True)
        self.gap = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.head = torch.nn.Linear(base * 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [B, 1, n_mels, time].

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


def load_cnn(model_path: str, device: torch.device) -> Tuple[torch.nn.Module, Dict]:
    """
    Load a trained CNN and its configuration.

    Args:
        model_path: Path to the CNN checkpoint (.pt).
        device: Torch device to load the model onto.

    Returns:
        Tuple of (model, config dictionary).
    """
    ckpt = torch.load(model_path, map_location=device)
    model = SmallResNet().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["config"]


def list_audio_files(
    dir_path: Path,
    exts: Tuple[str, ...],
    recursive: bool = True,
) -> List[Path]:
    """
    List audio files under a directory.

    Args:
        dir_path: Directory to search.
        exts: Allowed file extensions.
        recursive: Whether to search recursively.

    Returns:
        List of audio file paths.
    """
    pattern = "**/*" if recursive else "*"
    return [
        p for p in dir_path.glob(pattern)
        if p.is_file() and p.suffix.lower() in exts
    ]


def to_mono(wav: torch.Tensor) -> torch.Tensor:
    """
    Convert waveform to mono.

    Args:
        wav: Waveform tensor [C, T] or [T].

    Returns:
        Mono waveform [1, T].
    """
    if wav.dim() == 1:
        return wav.unsqueeze(0)
    if wav.size(0) > 1:
        return wav.mean(dim=0, keepdim=True)
    return wav


def resample_if_needed(
    wav: torch.Tensor,
    orig_sr: int,
    target_sr: int,
) -> torch.Tensor:
    """
    Resample waveform if sample rate differs.

    Args:
        wav: Waveform tensor [1, T].
        orig_sr: Original sample rate.
        target_sr: Target sample rate.

    Returns:
        Resampled waveform.
    """
    if orig_sr == target_sr:
        return wav
    return torchaudio.transforms.Resample(orig_sr, target_sr)(wav)


def center_crop_or_pad(wav: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    Center-crop or zero-pad waveform to target length.

    Args:
        wav: Waveform tensor [1, T].
        target_len: Target length in samples.

    Returns:
        Waveform tensor [1, target_len].
    """
    t = wav.size(-1)
    if t > target_len:
        start = (t - target_len) // 2
        return wav[:, start:start + target_len]
    if t < target_len:
        return torch.nn.functional.pad(wav, (0, target_len - t))
    return wav


def compute_silence_ratio(wav: torch.Tensor) -> float:
    """
    Estimate silence ratio using amplitude thresholding.

    Args:
        wav: Waveform tensor [1, T].

    Returns:
        Fraction of samples considered silent.
    """
    x = wav.abs().squeeze(0)
    thr = max(0.005, float(x.std()) * 0.15)
    return float((x < thr).float().mean())


def build_mel_transforms(cfg: Dict, device: torch.device):
    """
    Build mel spectrogram and amplitude-to-dB transforms.

    Args:
        cfg: CNN configuration dictionary.
        device: Torch device.

    Returns:
        Tuple of (MelSpectrogram, AmplitudeToDB).
    """
    melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg["sample_rate"],
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
        win_length=cfg["win_length"],
        f_min=cfg["f_min"],
        f_max=cfg["f_max"],
        n_mels=cfg["n_mels"],
        power=2.0,
    ).to(device)

    amp_to_db = torchaudio.transforms.AmplitudeToDB(
        stype="power", top_db=80
    ).to(device)

    return melspec, amp_to_db


@torch.no_grad()
def cnn_score_clip(
    model: torch.nn.Module,
    melspec,
    amp_to_db,
    wav_clip: torch.Tensor,
) -> float:
    """
    Compute CNN spoof probability for a single audio clip.

    Args:
        model: Trained CNN.
        melspec: MelSpectrogram transform.
        amp_to_db: AmplitudeToDB transform.
        wav_clip: Waveform clip [1, T].

    Returns:
        Spoof probability in [0, 1].
    """
    mel = melspec(wav_clip)
    mel_db = amp_to_db(mel)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    logits = model(mel_db.unsqueeze(0))
    return float(torch.sigmoid(logits).item())


def segment_start_positions(
    total_len: int,
    clip_len: int,
    n_segments: int,
) -> List[int]:
    """
    Compute evenly spaced segment start positions.

    Args:
        total_len: Total waveform length in samples.
        clip_len: Clip length in samples.
        n_segments: Number of segments.

    Returns:
        List of start indices.
    """
    if total_len <= clip_len:
        return [0] * n_segments
    if n_segments == 1:
        return [(total_len - clip_len) // 2]
    max_start = total_len - clip_len
    return [
        int(round(i * max_start / (n_segments - 1)))
        for i in range(n_segments)
    ]


@torch.no_grad()
def extract_features(
    audio_path: str,
    label: int,
    model: torch.nn.Module,
    cfg: Dict,
    melspec,
    amp_to_db,
    device: torch.device,
    clip_seconds: float,
    n_segments: int,
) -> Tuple[np.ndarray, int]:
    """
    Extract aggregated CNN features for a single audio file.

    Args:
        audio_path: Path to audio file.
        label: Ground-truth label (0=real, 1=fake).
        model: Trained CNN.
        cfg: CNN configuration.
        melspec: MelSpectrogram transform.
        amp_to_db: AmplitudeToDB transform.
        device: Torch device.
        clip_seconds: Segment duration in seconds.
        n_segments: Number of segments to sample.

    Returns:
        Tuple of (feature vector, label).
    """
    wav, sr = torchaudio.load(audio_path)
    wav = to_mono(wav).to(device)
    wav = resample_if_needed(wav, sr, cfg["sample_rate"])

    total_len = wav.size(-1)
    clip_len = int(cfg["sample_rate"] * clip_seconds)
    starts = segment_start_positions(total_len, clip_len, n_segments)

    scores = []
    for s in starts:
        clip = center_crop_or_pad(wav[:, s:s + clip_len], clip_len)
        scores.append(cnn_score_clip(model, melspec, amp_to_db, clip))

    scores = np.asarray(scores, dtype=np.float32)

    feats = np.array(
        [
            np.median(scores),
            np.max(scores),
            np.var(scores),
            total_len / cfg["sample_rate"],
            compute_silence_ratio(wav),
        ],
        dtype=np.float32,
    )
    return feats, label

def eval_at_threshold(y_true: np.ndarray, probs: np.ndarray, thresh: float):
    """
    Evaluate classification metrics at a given threshold.

    Args:
        y_true (np.ndarray): True binary labels.
        probs (np.ndarray): Predicted probabilities.
        thresh (float): Threshold for classification.

    Returns:
        _type_: _description_
    """
    preds = (probs >= thresh).astype(int)
    real_mask = y_true == 0
    fake_mask = y_true == 1
    real_acc = float((preds[real_mask] == 0).mean()) if real_mask.any() else float("nan")
    fake_acc = float((preds[fake_mask] == 1).mean()) if fake_mask.any() else float("nan")
    overall_acc = float((preds == y_true).mean()) if y_true.size else float("nan")
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    return {
        "threshold": float(thresh),
        "real_accuracy": real_acc,
        "fake_accuracy": fake_acc,
        "overall_accuracy": overall_acc,
        "overall_precision": precision,
        "confusion_matrix": confusion_matrix(y_true, preds, labels=[0, 1]).tolist(),
        "confusion_matrix_labels": ["real", "fake"],
    }


def eval_top_k_capture(y_true: np.ndarray, probs: np.ndarray, review_frac: float):
    """Evaluate top-k capture metrics.

    Args:
        y_true (np.ndarray): True binary labels.
        probs (np.ndarray): Predicted probabilities.
        review_frac (float): Fraction of samples to review.

    Returns:
        _type_: _description_
    """
    n = len(y_true)
    k = max(1, int(round(n * review_frac)))
    idx = np.argsort(-probs)[:k]
    reviewed = y_true[idx]
    fake_total = int((y_true == 1).sum())
    fake_caught = int((reviewed == 1).sum())
    catch_rate = float(fake_caught / fake_total) if fake_total else float("nan")
    reviewed_fake_rate = float((reviewed == 1).mean()) if len(reviewed) else float("nan")
    return {
        "review_frac": float(review_frac),
        "n_total": int(n),
        "n_review": int(k),
        "fake_total": int(fake_total),
        "fake_caught": int(fake_caught),
        "fake_catch_rate": catch_rate,
        "reviewed_fake_rate": reviewed_fake_rate,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn_model", required=True)
    ap.add_argument("--calibrator", required=True, help="Path to .joblib produced by train_aggregator_lr.py")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--clip_seconds", type=float, default=4.0)
    ap.add_argument("--n_segments", type=int, default=6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--exts", default=".wav,.flac,.mp3,.ogg,.m4a")
    ap.add_argument("--thresholds", default="0.3,0.5,0.7,0.85")
    args = ap.parse_args()

    device = torch.device(args.device)
    model, cfg = load_cnn(args.cnn_model, device=device)
    melspec, amp_to_db = build_mel_transforms(cfg, device=device)

    bundle = joblib.load(args.calibrator)
    agg = bundle["aggregator"] if isinstance(bundle, dict) and "aggregator" in bundle else bundle

    data_dir = Path(args.data_dir)
    exts = tuple(e.strip().lower() for e in args.exts.split(",") if e.strip())

    real_files = list_audio_files(data_dir / "real", exts=exts, recursive=True)
    fake_files = list_audio_files(data_dir / "fake", exts=exts, recursive=True)

    if not real_files or not fake_files:
        raise RuntimeError(f"Need files under {data_dir/'real'} and {data_dir/'fake'}; got real={len(real_files)} fake={len(fake_files)}")

    files = [(str(p), 0) for p in real_files] + [(str(p), 1) for p in fake_files]

    X = []
    y = []
    for p, lab in tqdm(files, desc="Extracting features"):
        feats, lab = extract_features(
            p, lab, model, cfg, melspec, amp_to_db, device, args.clip_seconds, args.n_segments
        )
        X.append(feats)
        y.append(lab)

    X = np.stack(X, axis=0)
    y = np.array(y, dtype=np.int64)

    probs = agg.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else float("nan")

    thresholds = [float(t) for t in args.thresholds.split(",") if t.strip()]
    threshold_metrics = [eval_at_threshold(y, probs, t) for t in thresholds]
    topk_metrics = [eval_top_k_capture(y, probs, f) for f in (0.05, 0.10, 0.20)]

    report = {
        "auc": float(auc),
        "n_total": int(len(y)),
        "feature_names": ["cnn_median", "cnn_max", "cnn_var", "total_seconds", "silence_ratio"],
        "threshold_metrics": threshold_metrics,
        "topk_metrics": topk_metrics,
    }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()