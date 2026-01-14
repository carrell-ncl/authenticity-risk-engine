#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

# Audio I/O
import soundfile as sf

# If you use torch for SmallResNet
import torch
import torch.nn.functional as F

from src.audio_det.inference.scorer import AudioSpoofScorer, ScorerConfig


# -----------------------------
# Utilities: audio loading
# -----------------------------
def load_audio_any(
    audio_bytes: bytes,
    target_sr: int = 16000,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Load audio from bytes (wav/flac/ogg/etc. depending on libsndfile support),
    convert to mono float32 in [-1,1], resample to target_sr if needed.

    Returns (y, sr).
    """
    data, sr = sf.read(io.BytesIO(audio_bytes), always_2d=True)
    # data shape: (n_samples, n_channels)
    if data.size == 0:
        return np.array([], dtype=np.float32), target_sr

    # Convert to float32
    y = data.astype(np.float32)

    # Mono
    if mono and y.shape[1] > 1:
        y = np.mean(y, axis=1, keepdims=True)

    y = y[:, 0]  # (n_samples,)

    # Resample if needed (avoid librosa dependency; use scipy if available)
    if sr != target_sr:
        try:
            from scipy.signal import resample_poly
            # rational approximation for resample
            g = math.gcd(sr, target_sr)
            up = target_sr // g
            down = sr // g
            y = resample_poly(y, up, down).astype(np.float32)
            sr = target_sr
        except Exception as e:
            raise RuntimeError(
                f"Resampling requires scipy. Install scipy or provide audio at {target_sr}Hz. "
                f"Original sr={sr}. Error={e}"
            )
    return y, sr


def load_audio_path(path: str, target_sr: int = 16000, mono: bool = True) -> Tuple[np.ndarray, int]:
    with open(path, "rb") as f:
        b = f.read()
    return load_audio_any(b, target_sr=target_sr, mono=mono)


# -----------------------------
# Segment sampling
# -----------------------------
def sample_fixed_segments(
    y: np.ndarray,
    sr: int,
    clip_seconds: float = 4.0,
    n_segments: int = 6,
) -> List[Tuple[int, int]]:
    """
    Sample n_segments fixed-length windows spread across the full audio duration.
    If the audio is shorter than clip_seconds, return a single padded segment.
    """
    if y.size == 0:
        return []

    clip_len = int(round(clip_seconds * sr))
    n = int(y.shape[0])

    if n <= clip_len:
        return [(0, n)]  # caller can pad if needed

    # Evenly spaced start times
    # Use linspace of start indices from 0..(n-clip_len)
    max_start = n - clip_len
    starts = np.linspace(0, max_start, num=n_segments, dtype=int)
    segs = [(int(s), int(s) + clip_len) for s in starts]
    return segs


def pad_or_trim(seg: np.ndarray, target_len: int) -> np.ndarray:
    if seg.shape[0] == target_len:
        return seg
    if seg.shape[0] > target_len:
        return seg[:target_len]
    out = np.zeros((target_len,), dtype=np.float32)
    out[: seg.shape[0]] = seg
    return out


def silence_ratio(
    y: np.ndarray,
    sr: int,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    db_threshold: float = -35.0,
) -> float:
    """
    Simple energy-based silence ratio over frames.
    """
    if y.size == 0:
        return 1.0
    frame = int(round(frame_ms * sr / 1000.0))
    hop = int(round(hop_ms * sr / 1000.0))
    if frame <= 0 or hop <= 0 or y.size < frame:
        return 1.0

    eps = 1e-10
    n_frames = 1 + (y.size - frame) // hop
    silent = 0
    for i in range(n_frames):
        s = i * hop
        e = s + frame
        fr = y[s:e]
        rms = np.sqrt(np.mean(fr * fr) + eps)
        db = 20.0 * np.log10(rms + eps)
        if db < db_threshold:
            silent += 1
    return float(silent / max(1, n_frames))


# -----------------------------
# Log-mel spectrogram (minimal, torch-based)
# -----------------------------
def log_mel_spectrogram(
    y: np.ndarray,
    sr: int,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
    n_mels: int = 80,
    fmin: float = 20.0,
    fmax: Optional[float] = None,
) -> torch.Tensor:
    """
    Returns torch tensor shape: (1, n_mels, time)
    Uses torch.stft + mel filter bank via librosa if available, else torchaudio.
    """
    if fmax is None:
        fmax = float(sr / 2)

    y_t = torch.from_numpy(y.astype(np.float32)).unsqueeze(0)  # (1, T)

    window = torch.hann_window(win_length)
    stft = torch.stft(
        y_t,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )
    power = (stft.abs() ** 2.0).squeeze(0)  # (freq, time)

    # Mel filterbank
    try:
        import torchaudio
        fb = torchaudio.functional.melscale_fbanks(
            n_freqs=power.shape[0],
            f_min=fmin,
            f_max=fmax,
            n_mels=n_mels,
            sample_rate=sr,
            norm="slaney",
            mel_scale="htk",
        )  # (freq, mels)
        mel = torch.matmul(fb.T, power)  # (mels, time)
    except Exception:
        # Fallback to librosa to build fbanks (works if installed)
        import librosa
        fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
        fb_t = torch.from_numpy(fb.astype(np.float32))  # (mels, freq)
        mel = torch.matmul(fb_t, power)  # (mels, time)

    mel = torch.clamp(mel, min=1e-10)
    logmel = torch.log(mel)
    return logmel.unsqueeze(0)  # (1, mels, time)


# -----------------------------
# Model bundle
# -----------------------------
@dataclass
class ScoreConfig:
    sr: int = 16000
    clip_seconds: float = 4.0
    n_segments: int = 6
    # risk tier thresholds (example defaults)
    low_thr: float = 0.3
    high_thr: float = 0.7
    # uncertainty threshold on |pA - pB| (if you later add dual calibrators)
    uncertainty_thr: float = 0.2


class SpoofScorer:
    """
    Loads:
      - CNN torch model checkpoint (SmallResNet)
      - Optional sklearn LR calibrator joblib (expects features in this order):
            ["cnn_median","cnn_max","cnn_var","total_seconds","silence_ratio"]
    Provides:
      - score_audio_bytes(...)
      - score_audio_path(...)
    """

    def __init__(
        self,
        cnn_ckpt_path: str,
        calibrator_joblib_path: Optional[str] = None,
        device: Optional[str] = None,
        config: Optional[ScoreConfig] = None,
    ):
        self.config = config or ScoreConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.cnn = self._load_cnn(cnn_ckpt_path).to(self.device).eval()
        self.calibrator = self._load_calibrator(calibrator_joblib_path) if calibrator_joblib_path else None

    def _load_cnn(self, path: str) -> torch.nn.Module:
        """
        Adjust this loader to your actual SmallResNet definition / checkpoint format.
        Supported patterns:
          - torch.jit.load(path) if you exported TorchScript
          - torch.load(state_dict) into your model class
        """
        if path.endswith(".pt") or path.endswith(".pth"):
            obj = torch.load(path, map_location="cpu")
            # If you saved a TorchScript model
            if isinstance(obj, torch.jit.ScriptModule):
                return obj

            # If you saved {"model_state": ..., "model_ctor": ...} etc.
            # You MUST adapt the below to your codebase.
            if isinstance(obj, dict) and "state_dict" in obj:
                state = obj["state_dict"]
            elif isinstance(obj, dict) and "model_state_dict" in obj:
                state = obj["model_state_dict"]
            elif isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
                # possibly a raw state_dict
                state = obj
            else:
                raise ValueError(f"Unrecognized checkpoint format for {path}")

            model = SmallResNet(n_mels=80)  # adapt signature
            model.load_state_dict(state, strict=False)
            return model

        # TorchScript fallback if you use .jit
        return torch.jit.load(path, map_location="cpu")

    def _load_calibrator(self, path: str):
        obj = joblib.load(path)
        # allow either direct sklearn model or {"pipeline": ...}
        if isinstance(obj, dict) and "pipeline" in obj:
            return obj["pipeline"]
        return obj

    @torch.no_grad()
    def _cnn_segment_prob(self, seg: np.ndarray, sr: int) -> float:
        """
        Computes spoof prob for one segment.
        Assumes CNN outputs logits for binary class or single logit.
        Adapt if your model outputs different structure.
        """
        # log-mel
        X = log_mel_spectrogram(seg, sr=sr)  # (1, mels, time)
        X = X.to(self.device)

        out = self.cnn(X)
        # handle common output styles
        if isinstance(out, (tuple, list)):
            out = out[0]
        out = out.squeeze()

        # If out has 2 logits (real/fake)
        if out.ndim == 1 and out.shape[0] == 2:
            prob_fake = float(F.softmax(out, dim=0)[1].item())
            return prob_fake

        # If single logit
        prob_fake = float(torch.sigmoid(out).item())
        return prob_fake

    def _aggregate_features(
        self,
        seg_probs: List[float],
        total_seconds: float,
        sil_ratio: float,
    ) -> Dict[str, float]:
        arr = np.array(seg_probs, dtype=np.float32)
        return {
            "cnn_median": float(np.median(arr)) if arr.size else 0.0,
            "cnn_max": float(np.max(arr)) if arr.size else 0.0,
            "cnn_var": float(np.var(arr)) if arr.size else 0.0,
            "total_seconds": float(total_seconds),
            "silence_ratio": float(sil_ratio),
        }

    def _tier(self, p: float) -> str:
        if p >= self.config.high_thr:
            return "high"
        if p >= self.config.low_thr:
            return "medium"
        return "low"

    def score_audio_bytes(
        self,
        audio_bytes: bytes,
        filename: str = "audio",
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        y, sr = load_audio_any(audio_bytes, target_sr=self.config.sr, mono=True)
        total_seconds = float(len(y) / sr) if sr > 0 else 0.0

        if y.size == 0 or total_seconds < 0.25:
            return {
                "ok": False,
                "error": "Empty or too-short audio.",
                "filename": filename,
                "meta": {"total_seconds": total_seconds, "sr": sr},
            }

        segs = sample_fixed_segments(y, sr, clip_seconds=self.config.clip_seconds, n_segments=self.config.n_segments)
        clip_len = int(round(self.config.clip_seconds * sr))

        seg_probs: List[float] = []
        seg_meta: List[Dict[str, Any]] = []

        for (s, e) in segs:
            seg = y[s:e]
            seg = pad_or_trim(seg, clip_len)
            p = self._cnn_segment_prob(seg, sr)
            seg_probs.append(p)
            seg_meta.append({"start_s": round(s / sr, 3), "end_s": round((s + clip_len) / sr, 3), "p_fake": round(p, 6)})

        sil_ratio = silence_ratio(y, sr)

        feats = self._aggregate_features(seg_probs, total_seconds=total_seconds, sil_ratio=sil_ratio)
        cnn_score = feats["cnn_median"]  # your current cnn_agg = median

        calibrated_score = None
        if self.calibrator is not None:
            X = np.array([[feats["cnn_median"], feats["cnn_max"], feats["cnn_var"], feats["total_seconds"], feats["silence_ratio"]]], dtype=np.float32)
            calibrated_score = float(self.calibrator.predict_proba(X)[0, 1])

        # Choose output score:
        # If calibrator exists, use it; else use cnn_score
        score = calibrated_score if calibrated_score is not None else cnn_score

        # threshold decision (optional override)
        thr = float(threshold) if threshold is not None else None
        decision = None
        if thr is not None:
            decision = "fake" if score >= thr else "real"

        report = {
            "ok": True,
            "filename": filename,
            "score": round(float(score), 6),
            "tier": self._tier(float(score)),
            "decision": decision,
            "models": {
                "cnn": {"agg": "median", "n_segments": len(seg_probs), "clip_seconds": self.config.clip_seconds},
                "calibrator": bool(self.calibrator is not None),
            },
            "signals": {
                "cnn_only_score": round(float(cnn_score), 6),
                "calibrated_score": (round(float(calibrated_score), 6) if calibrated_score is not None else None),
                "cnn_median": round(feats["cnn_median"], 6),
                "cnn_max": round(feats["cnn_max"], 6),
                "cnn_var": round(feats["cnn_var"], 6),
                "silence_ratio": round(feats["silence_ratio"], 6),
                "total_seconds": round(feats["total_seconds"], 6),
            },
            "segments": seg_meta,
            "meta": {"sr": sr, "total_seconds": round(total_seconds, 6)},
        }
        return report

    def score_audio_path(self, path: str, threshold: Optional[float] = None) -> Dict[str, Any]:
        y, _ = load_audio_path(path, target_sr=self.config.sr, mono=True)
        with open(path, "rb") as f:
            b = f.read()
        return self.score_audio_bytes(b, filename=os.path.basename(path), threshold=threshold)


# -----------------------------
# CLI entrypoint
# -----------------------------
def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="Path to audio file (.wav/.flac/...)")
    ap.add_argument("--cnn", required=True, help="Path to CNN checkpoint (.pt/.pth or torchscript)")
    ap.add_argument("--calibrator", default=None, help="Optional joblib LR calibrator")
    ap.add_argument("--threshold", type=float, default=None, help="Optional decision threshold for fake/real")
    args = ap.parse_args()

    scorer = SpoofScorer(
        cnn_ckpt_path=args.cnn,
        calibrator_joblib_path=args.calibrator,
    )
    report = scorer.score_audio_path(args.audio, threshold=args.threshold)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
