#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import logging

# Configure logger
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

import joblib
import librosa
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SR = 16000

def load_audio(path: str, sr: int = SR) -> np.ndarray:
    """Load audio file as mono float32 array.

    Args:
        path (str): Path to audio file.
        sr (int, optional): Sampling rate. Defaults to SR.

    Returns:
        np.ndarray: Audio signal as float32 array.
    """
    y, _sr = librosa.load(path, sr=sr, mono=True)
    # Avoid weird edge cases
    if y is None or len(y) < sr // 2:
        return np.array([], dtype=np.float32)
    return y.astype(np.float32)

def energy_vad_segments(y: np.ndarray, sr: int = SR,
                        top_db: int = 25,
                        min_seg_s: float = 1.0,
                        max_seg_s: float = 6.0):
    """Segment audio using energy-based VAD and chunking.

    Args:
        y (np.ndarray): Audio signal.
        sr (int, optional): Sampling rate. Defaults to SR.
        top_db (int, optional): Threshold for silence detection. Defaults to 25.
        min_seg_s (float, optional): Minimum segment length in seconds. Defaults to 1.0.
        max_seg_s (float, optional): Maximum segment length in seconds. Defaults to 6.0.

    Returns:
        list: List of tuples (start_sample, end_sample) for each segment.
    """
    if len(y) == 0:
        return []

    intervals = librosa.effects.split(y, top_db=top_db)
    segs = []
    min_len = int(min_seg_s * sr)
    max_len = int(max_seg_s * sr)

    for (s, e) in intervals:
        if e - s < min_len:
            continue
        # Chunk into max_seg_s windows for stability
        cur = s
        while cur < e:
            end = min(cur + max_len, e)
            if end - cur >= min_len:
                segs.append((cur, end))
            cur = end
    return segs

def segment_features(seg: np.ndarray, sr: int = SR) -> np.ndarray:
    """Extract handcrafted features per audio segment for a baseline model.

    Args:
        seg (np.ndarray): Audio segment.
        sr (int, optional): Sampling rate. Defaults to SR.

    Returns:
        np.ndarray: Feature vector for the segment.
    """
    # MFCCs
    mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=20)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    # Delta MFCCs
    dm = librosa.feature.delta(mfcc)
    dm_mean = dm.mean(axis=1)
    dm_std = dm.std(axis=1)

    # Spectral features
    centroid = librosa.feature.spectral_centroid(y=seg, sr=sr).mean()
    bandwidth = librosa.feature.spectral_bandwidth(y=seg, sr=sr).mean()
    flatness = librosa.feature.spectral_flatness(y=seg).mean()
    rolloff = librosa.feature.spectral_rolloff(y=seg, sr=sr, roll_percent=0.85).mean()
    zcr = librosa.feature.zero_crossing_rate(seg).mean()

    # Energy dynamics (over-smooth speech can be synthetic)
    rms = librosa.feature.rms(y=seg).flatten()
    rms_mean = float(rms.mean())
    rms_std = float(rms.std())
    rms_p95 = float(np.percentile(rms, 95))
    rms_p05 = float(np.percentile(rms, 5))

    # Simple pitch proxy via yin (can fail on noisy audio; handle gracefully)
    try:
        f0 = librosa.yin(seg, fmin=50, fmax=400, sr=sr)
        f0 = f0[np.isfinite(f0)]
        f0_mean = float(np.mean(f0)) if len(f0) else 0.0
        f0_std = float(np.std(f0)) if len(f0) else 0.0
    except Exception:
        f0_mean, f0_std = 0.0, 0.0

    feats = np.concatenate([
        mfcc_mean, mfcc_std, dm_mean, dm_std,
        np.array([centroid, bandwidth, flatness, rolloff, zcr,
                  rms_mean, rms_std, rms_p95, rms_p05,
                  f0_mean, f0_std], dtype=np.float32)
    ]).astype(np.float32)
    return feats

def file_features(path: str, sr: int = SR) -> tuple[np.ndarray, dict]:
    """Aggregate segment features into a single file vector using robust statistics.

    Args:
        path (str): Path to audio file.
        sr (int, optional): Sampling rate. Defaults to SR.

    Returns:
        tuple: (feature_vector as np.ndarray, meta as dict with speech_seconds and n_segments)
    """
    y = load_audio(path, sr=sr)
    if len(y) == 0:
        return np.array([], dtype=np.float32), {"speech_seconds": 0.0, "n_segments": 0}

    segs = energy_vad_segments(y, sr=sr)
    if not segs:
        return np.array([], dtype=np.float32), {"speech_seconds": 0.0, "n_segments": 0}

    seg_feat_list = []
    speech_samples = 0
    for (s, e) in segs:
        seg = y[s:e]
        speech_samples += (e - s)
        seg_feat_list.append(segment_features(seg, sr=sr))

    F = np.stack(seg_feat_list, axis=0)  # [n_seg, n_feat]

    # Robust aggregate: median + IQR
    med = np.median(F, axis=0)
    q75 = np.percentile(F, 75, axis=0)
    q25 = np.percentile(F, 25, axis=0)
    iqr = (q75 - q25)

    agg = np.concatenate([med, iqr]).astype(np.float32)

    meta = {
        "speech_seconds": float(speech_samples / sr),
        "n_segments": int(len(segs))
    }
    return agg, meta

def collect_dataset(data_dir: str):
    """Collect features and labels from all audio files in the dataset.

    Args:
        data_dir (str): Path to dataset directory containing 'real/' and 'fake/' subfolders.

    Returns:
        tuple: (X as np.ndarray, y as np.ndarray, metas as list of dicts, paths as list of str)
    """
    data_dir = Path(data_dir)
    real_files = sorted([str(p) for p in (data_dir / "real").glob("*") if p.is_file()])
    fake_files = sorted([str(p) for p in (data_dir / "fake").glob("*") if p.is_file()])

    X, y, metas, paths = [], [], [], []

    for label, files in [(0, real_files), (1, fake_files)]:
        label_str = 'real' if label == 0 else 'fake'
        logger.info(f"Processing {len(files)} {label_str} files...")
        for idx, fp in enumerate(files, 1):
            feats, meta = file_features(fp)
            if feats.size == 0:
                logger.warning(f"Skipped file (no usable audio): {fp}")
                continue
            X.append(feats)
            y.append(label)
            metas.append(meta)
            paths.append(fp)
            if idx % 10 == 0 or idx == len(files):
                logger.info(f"Processed {idx}/{len(files)} {label_str} files")

    if not X:
        raise RuntimeError("No usable files found. Check your data folder and formats.")

    X = np.stack(X, axis=0)
    y = np.array(y, dtype=np.int64)
    return X, y, metas, paths

def main():
    """Main entry point for training the baseline audio classifier.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data", help="Folder containing real/ and fake/ subfolders")
    ap.add_argument("--out_model", default="models/audio_baseline.joblib")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logger.info("Collecting dataset features...")
    X, y, metas, paths = collect_dataset(args.data_dir)

    logger.info(f"Dataset collected: {X.shape[0]} samples, {X.shape[1]} features per sample.")

    logger.info("Splitting dataset into train and test sets...")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    logger.info(f"Train set: {X_tr.shape[0]} samples, Test set: {X_te.shape[0]} samples.")

    logger.info("Initializing and training classifier pipeline...")
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

    clf.fit(X_tr, y_tr)
    logger.info("Model training complete.")

    logger.info("Evaluating model on test set...")
    proba = clf.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, proba)

    logger.info(f"AUC: {auc:.4f}")
    logger.info("Classification report:\n" + classification_report(y_te, (proba >= 0.5).astype(int), digits=3))

    out_path = Path(args.out_model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": clf,
        "sr": SR,
        "feature_version": "v1_mfcc_spectral_pitch_median_iqr"
    }, out_path)

    logger.info(f"Saved model to: {out_path}")

if __name__ == "__main__":
    main()
