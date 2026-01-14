from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import io
import joblib
import numpy as np
import torch
import torchaudio


# ----------------------------
# Model definition (same as training)
# ----------------------------
class SmallResNet(torch.nn.Module):
    """Small residual CNN for spoof detection on log-mel spectrograms."""

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


# ----------------------------
# Shared utilities (copied from your eval script)
# ----------------------------
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


def load_cnn_model(model_path: str, device: torch.device) -> Tuple[torch.nn.Module, Dict]:
    ckpt = torch.load(model_path, map_location=device)
    cfg = ckpt["config"]
    model = SmallResNet(in_ch=1, base=32).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


def load_calibrator(calibrator_path: str):
    obj = joblib.load(calibrator_path)
    if isinstance(obj, dict) and "aggregator" in obj:
        return obj["aggregator"]
    return obj


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


def aggregate_cnn_probs(seg_probs: np.ndarray, agg: str = "median") -> float:
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


# ----------------------------
# Production scorer (API-friendly)
# ----------------------------
@dataclass
class ScorerConfig:
    n_segments: int = 6
    clip_seconds: float = 4.0
    cnn_agg: str = "median"
    low_thr: float = 0.3
    high_thr: float = 0.7


class AudioSpoofScorer:
    def __init__(
        self,
        cnn_model_path: str,
        calibrator_path: Optional[str] = None,
        device: str = "cpu",
        config: Optional[ScorerConfig] = None,
    ):
        self.device = torch.device(device)
        self.model, self.cfg = load_cnn_model(cnn_model_path, self.device)
        self.melspec, self.amp_to_db = build_transforms(self.cfg, self.device)
        self.calibrator = load_calibrator(calibrator_path) if calibrator_path else None

        self.config = config or ScorerConfig()
        # default clip_seconds to training cfg if not explicitly set
        if self.config.clip_seconds is None:
            self.config.clip_seconds = float(self.cfg.get("clip_seconds", 4.0))

    def tier(self, score: float) -> str:
        if score >= self.config.high_thr:
            return "high"
        if score >= self.config.low_thr:
            return "medium"
        return "low"

    def score_waveform(self, wav: torch.Tensor, sr: int, filename: str = "audio", threshold: Optional[float] = None) -> Dict[str, Any]:
        wav = to_mono(wav)

        # compute duration + silence on resampled waveform (same as eval)
        target_sr = int(self.cfg["sample_rate"])
        wav_rs = resample_if_needed(wav, sr, target_sr)
        total_seconds = float(wav_rs.size(-1) / target_sr) if target_sr else 0.0
        sil_ratio = compute_silence_ratio(wav_rs)

        if wav_rs.numel() == 0 or total_seconds < 0.25:
            return {
                "ok": False,
                "error": "Empty or too-short audio.",
                "filename": filename,
                "meta": {"total_seconds": total_seconds, "sr": target_sr},
            }

        seg_probs = score_segments_cnn(
            model=self.model,
            cfg=self.cfg,
            melspec=self.melspec,
            amp_to_db=self.amp_to_db,
            wav=wav,
            sr=sr,
            clip_seconds=float(self.config.clip_seconds),
            n_segments=int(self.config.n_segments),
            device=self.device,
        )

        cnn_score = aggregate_cnn_probs(seg_probs, self.config.cnn_agg)

        feats = compute_calibrator_features(seg_probs, total_seconds, sil_ratio)
        calibrated_score = None
        if self.calibrator is not None:
            calibrated_score = float(self.calibrator.predict_proba(feats.reshape(1, -1))[0, 1])

        score = calibrated_score if calibrated_score is not None else float(cnn_score)

        decision = None
        if threshold is not None:
            decision = "fake" if score >= float(threshold) else "real"

        return {
            "ok": True,
            "filename": filename,
            "score": round(float(score), 6),
            "tier": self.tier(float(score)),
            "decision": decision,
            "signals": {
                "cnn_only_score": round(float(cnn_score), 6),
                "calibrated_score": (round(float(calibrated_score), 6) if calibrated_score is not None else None),
                "cnn_median": round(float(np.median(seg_probs)), 6),
                "cnn_max": round(float(np.max(seg_probs)), 6),
                "cnn_var": round(float(np.var(seg_probs)), 6),
                "total_seconds": round(float(total_seconds), 6),
                "silence_ratio": round(float(sil_ratio), 6),
            },
            "segments": [round(float(p), 6) for p in seg_probs.tolist()],
            "meta": {
                "sr": int(target_sr),
                "clip_seconds": float(self.config.clip_seconds),
                "n_segments": int(self.config.n_segments),
                "cnn_agg": self.config.cnn_agg,
            },
            "model": {
                "cnn_model": "SmallResNet",
                "calibrator": bool(self.calibrator is not None),
            },
        }

    def score_file(self, path: str, threshold: Optional[float] = None) -> Dict[str, Any]:
        wav, sr = torchaudio.load(path)
        return self.score_waveform(wav, sr, filename=path.split("/")[-1], threshold=threshold)

    def score_bytes(self, audio_bytes: bytes, filename: str = "audio", threshold: Optional[float] = None) -> Dict[str, Any]:
        # torchaudio.load can read from file-like objects
        with io.BytesIO(audio_bytes) as bio:
            wav, sr = torchaudio.load(bio)
        return self.score_waveform(wav, sr, filename=filename, threshold=threshold)
