#!/usr/bin/env python3
"""
train_aggregator_lr.py

Train a logistic-regression *aggregator* on top of CNN inference outputs and
lightweight metadata features, then evaluate it using risk-oriented metrics.

Design intent:
- The CNN provides a strong, high-recall signal for synthetic audio.
- The aggregator learns a calibrated decision surface using:
    - CNN risk scores
    - evidence length
    - score stability across segments
    - simple acoustic metadata
- Logistic regression is used here as a *combiner and calibrator*, not as a
  standalone detector on handcrafted audio features.

Expected directory layout:
  eval_dir/
    real/   (*.wav, *.flac, ...)
    fake/   (*.wav, *.flac, ...)

Dependencies:
  torch
  torchaudio
  numpy
  scikit-learn
  joblib
  tqdm

Example usage:
  python train_aggregator_lr.py \
      --cnn_model models/audio_cnn_mel.pt \
      --data_dir path/to/eval_dir \
      --out models/agg_lr.joblib \
      --clip_seconds 4.0 \
      --n_segments 6
"""


import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import torch
import torchaudio
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


# -------------------------
# Model definition (must match training)
# -------------------------
class SmallResNet(torch.nn.Module):
    def __init__(self, in_ch: int = 1, base: int = 32):
        super().__init__()

        def block(cin, cout, stride=1):
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


def load_cnn(model_path: str, device: torch.device):
    ckpt = torch.load(model_path, map_location=device)
    cfg = ckpt["config"]
    model = SmallResNet(in_ch=1, base=32).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


# -------------------------
# Audio helpers
# -------------------------
def list_audio_files(dir_path: Path, exts=(".wav", ".flac", ".mp3", ".ogg", ".m4a"), recursive=True) -> List[Path]:
    exts = {e.lower() for e in exts}
    pattern = "**/*" if recursive else "*"
    files = [p for p in dir_path.glob(pattern) if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def to_mono(wav: torch.Tensor) -> torch.Tensor:
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav


def resample_if_needed(wav: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return wav
    return torchaudio.transforms.Resample(orig_sr, target_sr)(wav)


def center_crop_or_pad(wav: torch.Tensor, target_len: int) -> torch.Tensor:
    t = wav.size(-1)
    if t > target_len:
        start = (t - target_len) // 2
        return wav[:, start : start + target_len]
    if t < target_len:
        pad = target_len - t
        return torch.nn.functional.pad(wav, (0, pad), mode="constant", value=0.0)
    return wav


def compute_silence_ratio(wav: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Cheap silence proxy: fraction of samples below a small amplitude threshold.
    Not perfect, but useful as a meta-feature and very fast.
    """
    x = wav.abs().squeeze(0)
    thr = max(0.005, float(x.std().item()) * 0.15)  # adaptive-ish threshold
    return float((x < thr).float().mean().item())


def build_mel_transforms(cfg: Dict, device: torch.device):
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


@torch.no_grad()
def cnn_score_clip(model, melspec, amp_to_db, wav_clip: torch.Tensor) -> float:
    """
    wav_clip: [1, T] at target SR, already cropped/padded.
    Returns probability of spoof.
    """
    mel = melspec(wav_clip)
    mel_db = amp_to_db(mel)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    x = mel_db.unsqueeze(0)  # [1, 1, n_mels, time]
    logits = model(x)
    return float(torch.sigmoid(logits).item())


def segment_start_positions(total_len: int, clip_len: int, n_segments: int) -> List[int]:
    """
    Choose segment windows across the file. If file is shorter than clip_len, we'll pad anyway
    (start positions all 0).
    """
    if total_len <= clip_len:
        return [0] * n_segments
    if n_segments <= 1:
        return [(total_len - clip_len) // 2]
    # equally spaced windows
    max_start = total_len - clip_len
    return [int(round(i * max_start / (n_segments - 1))) for i in range(n_segments)]


@torch.no_grad()
def extract_aggregator_features(
    audio_path: str,
    model,
    cfg: Dict,
    melspec,
    amp_to_db,
    device: torch.device,
    clip_seconds: float,
    n_segments: int,
) -> Tuple[np.ndarray, Dict]:
    """
    Returns:
      features: np.array([cnn_median, cnn_max, cnn_var, speech_seconds, silence_ratio])
      meta: dict (durations, etc.)
    """
    wav, sr = torchaudio.load(audio_path)
    wav = to_mono(wav).to(device)

    target_sr = int(cfg["sample_rate"])
    wav = resample_if_needed(wav, sr, target_sr)

    total_len = wav.size(-1)
    total_seconds = float(total_len / target_sr)

    clip_len = int(target_sr * clip_seconds)

    starts = segment_start_positions(total_len, clip_len, n_segments)
    seg_scores = []
    for s in starts:
        clip = wav[:, s : s + clip_len]
        clip = center_crop_or_pad(clip, clip_len)
        seg_scores.append(cnn_score_clip(model, melspec, amp_to_db, clip))

    seg_scores = np.array(seg_scores, dtype=np.float32)

    cnn_median = float(np.median(seg_scores))
    cnn_max = float(np.max(seg_scores))
    cnn_var = float(np.var(seg_scores))
    sil_ratio = compute_silence_ratio(center_crop_or_pad(wav, min(total_len, clip_len)))

    feats = np.array([cnn_median, cnn_max, cnn_var, total_seconds, sil_ratio], dtype=np.float32)

    meta = {
        "audio_path": audio_path,
        "total_seconds": total_seconds,
        "segment_scores": seg_scores.tolist(),
        "cnn_median": cnn_median,
        "cnn_max": cnn_max,
        "cnn_var": cnn_var,
        "silence_ratio": sil_ratio,
    }
    return feats, meta


# -------------------------
# Evaluation helpers
# -------------------------
def eval_at_threshold(y_true: np.ndarray, probs: np.ndarray, thresh: float) -> Dict:
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


def eval_top_k_capture(y_true: np.ndarray, probs: np.ndarray, review_frac: float) -> Dict:
    """
    Insurance-style metric:
      "If we review the top X% highest-risk files, what fraction of fakes do we catch?"
    """
    n = len(y_true)
    k = max(1, int(round(n * review_frac)))
    idx = np.argsort(-probs)[:k]  # top risk
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


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn_model", required=True, help="Path to trained CNN .pt")
    ap.add_argument("--data_dir", required=True, help="Folder containing real/ and fake/ subfolders")
    ap.add_argument("--out", default="models/agg_lr.joblib", help="Where to save aggregator model")
    ap.add_argument("--clip_seconds", type=float, default=4.0, help="Clip length used for feature extraction")
    ap.add_argument("--n_segments", type=int, default=6, help="How many segments to sample per file for stability")
    ap.add_argument("--test_size", type=float, default=0.25, help="Holdout fraction for evaluation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--exts", default=".wav,.flac,.mp3,.ogg,.m4a", help="Comma-separated audio extensions")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    device = torch.device(args.device)

    print("Torch CUDA available:", torch.cuda.is_available())
    print("Using device:", device)


    model, cfg = load_cnn(args.cnn_model, device=device)
    melspec, amp_to_db = build_mel_transforms(cfg, device=device)

    data_dir = Path(args.data_dir)
    exts = tuple(e.strip().lower() for e in args.exts.split(",") if e.strip())

    real_files = list_audio_files(data_dir / "real", exts=exts, recursive=True)
    fake_files = list_audio_files(data_dir / "fake", exts=exts, recursive=True)

    if not real_files or not fake_files:
        raise RuntimeError(f"Need files under {data_dir/'real'} and {data_dir/'fake'}; got real={len(real_files)} fake={len(fake_files)}")

    files = [(str(p), 0) for p in real_files] + [(str(p), 1) for p in fake_files]
    rng.shuffle(files)

    paths = [p for p, _ in files]
    labels = np.array([y for _, y in files], dtype=np.int64)

    # Extract aggregator features
    X = []
    metas = []
    y = []
    for p, lab in tqdm(files, desc="Extracting features"):
        feats, meta = extract_aggregator_features(
            p,
            model=model,
            cfg=cfg,
            melspec=melspec,
            amp_to_db=amp_to_db,
            device=device,
            clip_seconds=args.clip_seconds,
            n_segments=args.n_segments,
        )
        X.append(feats)
        metas.append(meta)
        y.append(lab)

    X = np.stack(X, axis=0)
    y = np.array(y, dtype=np.int64)

    # Holdout split
    X_tr, X_te, y_tr, y_te, meta_tr, meta_te = train_test_split(
        X, y, metas, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # Train logistic regression aggregator (scaled)
    agg = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])
    agg.fit(X_tr, y_tr)

    # Evaluate
    probs = agg.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, probs) if len(np.unique(y_te)) > 1 else float("nan")

    # Report a few thresholds + top-k capture metrics
    threshold_metrics = [eval_at_threshold(y_te, probs, t) for t in (0.3, 0.5, 0.7, 0.85)]
    topk_metrics = [eval_top_k_capture(y_te, probs, f) for f in (0.05, 0.10, 0.20)]

    report = {
        "auc": float(auc),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "feature_names": ["cnn_median", "cnn_max", "cnn_var", "total_seconds", "silence_ratio"],
        "threshold_metrics": threshold_metrics,
        "topk_metrics": topk_metrics,
    }

    print(json.dumps(report, indent=2))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "aggregator": agg,
            "cnn_model_path": args.cnn_model,
            "cnn_cfg": cfg,
            "clip_seconds": args.clip_seconds,
            "n_segments": args.n_segments,
            "feature_names": report["feature_names"],
            "report": report,
        },
        out_path
    )

    print(f"\nSaved aggregator to: {out_path}")


if __name__ == "__main__":
    main()
