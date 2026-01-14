#!/usr/bin/env python3
"""
evaluate_cnn_and_dual_calibrators.py

Evaluate:
1) CNN-only
2) Calibrator A
3) Calibrator B
4) Combined score (mean/max/min)
5) Uncertainty metrics and "domain shift" gating

Directory format:
  data_dir/
    real/
    fake/

Example:
python src/inference/evaluate_cnn_and_dual_calibrators.py \
  --cnn_model models/cnn/audio_cnn_mel.pt \
  --data_dir data/audio/processed/asvspoof_2021_df \
  --calibrators models/calibrators/agg_lr_agvspoof.joblib,models/calibrators/agg_lr_real_or_fake.joblib \
  --combine mean \
  --uncertainty_threshold 0.25 \
  --device cpu \
  --sample_per_class 2000 \
  --seed 42

"""

from __future__ import annotations

import argparse
import json
import random
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
    def __init__(self, in_ch: int = 1, base: int = 32):
        super().__init__()

        def block(cin: int, cout: int, stride: int = 1) -> torch.nn.Module:
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
    ckpt = torch.load(model_path, map_location=device)
    cfg = ckpt["config"]
    model = SmallResNet(in_ch=1, base=32).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


def build_transforms(cfg: Dict, device: torch.device):
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


def list_audio_files(class_dir: Path, exts: Sequence[str], recursive: bool = True) -> List[Path]:
    exts_set = {e.lower() for e in exts}
    pattern = "**/*" if recursive else "*"
    files = [p for p in class_dir.glob(pattern) if p.is_file() and p.suffix.lower() in exts_set]
    return sorted(files)


def to_mono(wav: torch.Tensor) -> torch.Tensor:
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav


def resample_if_needed(wav: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    if sr == target_sr:
        return wav
    return torchaudio.transforms.Resample(sr, target_sr)(wav)


def center_crop_or_pad(wav: torch.Tensor, target_len: int) -> torch.Tensor:
    t = wav.size(-1)
    if t > target_len:
        start = (t - target_len) // 2
        return wav[:, start : start + target_len]
    if t < target_len:
        return torch.nn.functional.pad(wav, (0, target_len - t), mode="constant", value=0.0)
    return wav


def segment_start_positions(total_len: int, clip_len: int, n_segments: int) -> List[int]:
    if n_segments <= 1:
        return [max(0, (total_len - clip_len) // 2)]
    if total_len <= clip_len:
        return [0] * n_segments
    max_start = total_len - clip_len
    return [int(round(i * max_start / (n_segments - 1))) for i in range(n_segments)]


def compute_silence_ratio(wav: torch.Tensor) -> float:
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
    agg = agg.lower()
    if seg_probs.size == 0:
        return float("nan")
    if agg == "median":
        return float(np.median(seg_probs))
    if agg == "mean":
        return float(np.mean(seg_probs))
    if agg == "max":
        return float(np.max(seg_probs))
    raise ValueError("agg must be one of: median, mean, max")


def compute_calibrator_features(seg_probs: np.ndarray, total_seconds: float, silence_ratio: float) -> np.ndarray:
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
    if sample_per_class is None:
        return real_files, fake_files
    rng = random.Random(seed)
    real = real_files if sample_per_class >= len(real_files) else rng.sample(real_files, sample_per_class)
    fake = fake_files if sample_per_class >= len(fake_files) else rng.sample(fake_files, sample_per_class)
    return sorted(real), sorted(fake)


def load_calibrator(path: str):
    obj = joblib.load(path)
    if isinstance(obj, dict) and "aggregator" in obj:
        return obj["aggregator"]
    return obj


def parse_thresholds(s: str) -> List[float]:
    out: List[float] = []
    for t in s.split(","):
        t = t.strip()
        if t:
            out.append(float(t))
    return out if out else [0.5]


def combine_probs(p1: np.ndarray, p2: np.ndarray, method: str) -> np.ndarray:
    method = method.lower()
    if method == "mean":
        return 0.5 * (p1 + p2)
    if method == "max":
        return np.maximum(p1, p2)
    if method == "min":
        return np.minimum(p1, p2)
    raise ValueError("combine must be one of: mean, max, min")


def uncertainty_summary(u: np.ndarray) -> Dict:
    if u.size == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p90": float("nan"), "p95": float("nan"), "p99": float("nan")}
    return {
        "mean": float(np.mean(u)),
        "p50": float(np.percentile(u, 50)),
        "p90": float(np.percentile(u, 90)),
        "p95": float(np.percentile(u, 95)),
        "p99": float(np.percentile(u, 99)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn_model", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument(
        "--calibrators",
        required=True,
        help="Comma-separated calibrator paths: calA,calB",
    )
    ap.add_argument("--combine", default="mean", choices=["mean", "max", "min"])
    ap.add_argument(
        "--uncertainty_threshold",
        type=float,
        default=0.25,
        help="If |pA - pB| > threshold, consider it domain-shifted.",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--exts", default=".flac,.wav")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--sample_per_class", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clip_seconds", type=float, default=None)
    ap.add_argument("--n_segments", type=int, default=6)
    ap.add_argument("--cnn_agg", default="median", choices=["median", "mean", "max"])
    ap.add_argument("--thresholds", default="0.3,0.5,0.7,0.85")
    ap.add_argument("--report_out", default=None)
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

    cal_paths = [p.strip() for p in args.calibrators.split(",") if p.strip()]
    if len(cal_paths) != 2:
        raise ValueError("--calibrators must contain exactly two comma-separated paths.")
    calA = load_calibrator(cal_paths[0])
    calB = load_calibrator(cal_paths[1])

    y_true: List[int] = []
    cnn_probs: List[float] = []
    feats_list: List[np.ndarray] = []

    for p, lab in files:
        wav, sr = torchaudio.load(str(p))
        wav = to_mono(wav)

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

        y_true.append(lab)
        cnn_probs.append(aggregate_cnn_probs(seg_probs, args.cnn_agg))
        feats_list.append(compute_calibrator_features(seg_probs, total_seconds, silence_ratio))

    y_true_np = np.asarray(y_true, dtype=np.int64)
    cnn_probs_np = np.asarray(cnn_probs, dtype=np.float32)
    X = np.stack(feats_list, axis=0).astype(np.float32)

    # Individual calibrators
    pA = calA.predict_proba(X)[:, 1].astype(np.float32)
    pB = calB.predict_proba(X)[:, 1].astype(np.float32)

    # Combined + uncertainty
    p_mix = combine_probs(pA, pB, args.combine).astype(np.float32)
    u = np.abs(pA - pB).astype(np.float32)
    domain_shift = (u > float(args.uncertainty_threshold)).astype(np.int64)

    # Gated score (policy):
    # - if domain_shift: fall back to CNN score for ranking/classification
    # - else: use combined calibrated score
    # p_gated = np.where(domain_shift == 1, cnn_probs_np, p_mix).astype(np.float32)

    # NEW: pick calibrator closer to CNN score (per file)
    pick_A = np.abs(pA - cnn_probs_np) <= np.abs(pB - cnn_probs_np)
    print("DEBUG pick_A_rate:", float(pick_A.mean()))
    p_pick = np.where(pick_A, pA, pB).astype(np.float32)

    # Gated score: fallback to CNN on disagreement, else use picked calibrator
    p_gated = np.where(domain_shift == 1, cnn_probs_np, p_pick).astype(np.float32)


    def pack_section(name: str, probs: np.ndarray) -> Dict:
        auc = float(roc_auc_score(y_true_np, probs)) if len(np.unique(y_true_np)) > 1 else float("nan")
        return {
            "name": name,
            "auc": auc,
            "threshold_metrics": [compute_metrics_at_threshold(y_true_np, probs, t) for t in thresholds],
            "topk_metrics": [compute_topk_metrics(y_true_np, probs, f) for f in (0.05, 0.10, 0.20)],
        }

    report: Dict = {
        "data_dir": str(data_dir),
        "n_total": int(len(y_true_np)),
        "n_real": int((y_true_np == 0).sum()),
        "n_fake": int((y_true_np == 1).sum()),
        "clip_seconds": float(clip_seconds),
        "n_segments": int(args.n_segments),
        "cnn_agg": args.cnn_agg,
        "thresholds": thresholds,
        "feature_names": ["cnn_median", "cnn_max", "cnn_var", "total_seconds", "silence_ratio"],
        "cnn_only": pack_section("cnn_only", cnn_probs_np),
        "calibrator_A": {"path": cal_paths[0], **pack_section("calibrator_A", pA)},
        "calibrator_B": {"path": cal_paths[1], **pack_section("calibrator_B", pB)},
        "combined": {"combine": args.combine, **pack_section("combined", p_mix)},
        "uncertainty": {
            "uncertainty_threshold": float(args.uncertainty_threshold),
            "summary": uncertainty_summary(u),
            "domain_shift_rate": float(domain_shift.mean()),
        },
        "gated": {
            "policy": "use CNN when |pA - pB| > threshold else use calibrator closest to CNN"
,
            **pack_section("gated", p_gated),
        },
    }

    print(json.dumps(report, indent=2))

    if args.report_out:
        out_path = Path(args.report_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
