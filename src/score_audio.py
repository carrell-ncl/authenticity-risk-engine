import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score
import random
import joblib
from collections import defaultdict
from sklearn.metrics import confusion_matrix

from src.train_baseline import file_features


def score_eval_directory(model_path, eval_dir, sr=16000, thresh=0.5, seed=None, sample_size=None):
    """Score all (or a sample of) audio files in 'real' and 'fake' subfolders and calculate per-class and overall accuracy.

    Args:
        model_path (str): Path to the trained model or pipeline (joblib file).
        eval_dir (str): Path to directory containing 'real' and 'fake' subfolders.
        sr (int, optional): Sampling rate for audio loading. Defaults to 16000.
        thresh (float, optional): Decision threshold for classifying as 'fake'. Defaults to 0.5.
        seed (int, optional): Random seed for reproducible sampling. Defaults to None.
        sample_size (int, optional): If set, randomly sample this many files from each class. Defaults to None (use all files).

    Returns:
        dict: Dictionary with per-class and overall accuracy and results.
    """
    model_obj = joblib.load(model_path)
    model = model_obj["pipeline"] if isinstance(model_obj, dict) and "pipeline" in model_obj else model_obj
    results = defaultdict(list)
    summary = {}
    y_true_all = []
    y_pred_all = []
    for label_str, label in [('real', 0), ('fake', 1)]:
        class_dir = os.path.join(eval_dir, label_str)
        if not os.path.isdir(class_dir):
            summary[f'{label_str}_accuracy'] = None
            summary[f'{label_str}_n'] = 0
            continue
        files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith('.flac') or f.endswith('.wav')]
        if sample_size is not None and sample_size < len(files):
            rng = random.Random(seed)
            files = rng.sample(files, sample_size)
        y_true = []
        y_pred = []
        for f in files:
            feats, _ = file_features(f, sr=sr)
            if feats.size == 0:
                continue
            proba = model.predict_proba([feats])[0, 1]
            pred = int(proba >= thresh)
            y_true.append(label)
            y_pred.append(pred)
            results[label_str].append((f, pred, proba))
            y_true_all.append(label)
            y_pred_all.append(pred)
        acc = accuracy_score(y_true, y_pred) if y_true else 0.0
        summary[f'{label_str}_accuracy'] = acc
        summary[f'{label_str}_n'] = len(y_true)
    summary['overall_accuracy'] = accuracy_score(y_true_all, y_pred_all) if y_true_all else 0.0
    summary['overall_precision'] = precision_score(y_true_all, y_pred_all) if y_true_all else 0.0
    summary['results'] = dict(results)
    summary['n_total'] = len(y_true_all)
    # Add confusion matrix
    if y_true_all and y_pred_all:
        cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])
        summary['confusion_matrix'] = cm.tolist()
        summary['confusion_matrix_labels'] = ['real', 'fake']
    else:
        summary['confusion_matrix'] = None
        summary['confusion_matrix_labels'] = ['real', 'fake']
    return summary
# Add function to score all audio files in a directory and calculate accuracy


def score_directory(model, directory, label, sr=16000, seed=None, sample_size=None):
    """Score all (or a sample of) audio files in a directory and calculate accuracy.

    Args:
        model: Trained model or pipeline with predict or predict_proba method.
        directory (str): Path to directory containing audio files.
        label (int): The ground truth label for all files in this directory (e.g., 0=real, 1=fake).
        sr (int, optional): Sampling rate for audio loading. Defaults to 16000.
        seed (int, optional): Random seed for reproducible sampling. Defaults to None.
        sample_size (int, optional): If set, randomly sample this many files from the directory. Defaults to None (use all files).

    Returns:
        float: Accuracy of the model on the directory.
        list: List of (filename, predicted_label, probability) tuples.
    """
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.flac')]
    if sample_size is not None and sample_size < len(files):
        rng = random.Random(seed)
        files = rng.sample(files, sample_size)
    y_true = []
    y_pred = []
    results = []
    for f in files:
        feats, _ = file_features(f, sr=sr)
        if feats.size == 0:
            continue
        proba = model.predict_proba([feats])[0, 1]
        pred = int(proba >= 0.5)
        y_true.append(label)
        y_pred.append(pred)
        results.append((f, pred, proba))
    acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    return acc, results
#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import joblib
import librosa
import numpy as np

SR_DEFAULT = 16000

def load_audio(path: str, sr: int) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    return (y.astype(np.float32) if y is not None else np.array([], dtype=np.float32))

def energy_vad_segments(y: np.ndarray, sr: int,
                        top_db: int = 25,
                        min_seg_s: float = 1.0,
                        max_seg_s: float = 6.0):
    if len(y) == 0:
        return []
    intervals = librosa.effects.split(y, top_db=top_db)
    segs = []
    min_len = int(min_seg_s * sr)
    max_len = int(max_seg_s * sr)
    for (s, e) in intervals:
        if e - s < min_len:
            continue
        cur = s
        while cur < e:
            end = min(cur + max_len, e)
            if end - cur >= min_len:
                segs.append((cur, end))
            cur = end
    return segs

def segment_features(seg: np.ndarray, sr: int) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=20)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    dm = librosa.feature.delta(mfcc)
    dm_mean = dm.mean(axis=1)
    dm_std = dm.std(axis=1)

    centroid = librosa.feature.spectral_centroid(y=seg, sr=sr).mean()
    bandwidth = librosa.feature.spectral_bandwidth(y=seg, sr=sr).mean()
    flatness = librosa.feature.spectral_flatness(y=seg).mean()
    rolloff = librosa.feature.spectral_rolloff(y=seg, sr=sr, roll_percent=0.85).mean()
    zcr = librosa.feature.zero_crossing_rate(seg).mean()

    rms = librosa.feature.rms(y=seg).flatten()
    rms_mean = float(rms.mean())
    rms_std = float(rms.std())
    rms_p95 = float(np.percentile(rms, 95))
    rms_p05 = float(np.percentile(rms, 5))

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

def file_features_and_segments(path: str, sr: int):
    y = load_audio(path, sr=sr)
    if len(y) == 0:
        return None, {"speech_seconds": 0.0, "n_segments": 0}, []

    segs = energy_vad_segments(y, sr=sr)
    if not segs:
        return None, {"speech_seconds": 0.0, "n_segments": 0}, []

    seg_feats = []
    seg_scores = []
    speech_samples = 0

    for (s, e) in segs:
        seg = y[s:e]
        speech_samples += (e - s)
        seg_feats.append(segment_features(seg, sr=sr))

    F = np.stack(seg_feats, axis=0)
    med = np.median(F, axis=0)
    q75 = np.percentile(F, 75, axis=0)
    q25 = np.percentile(F, 25, axis=0)
    iqr = (q75 - q25)
    agg = np.concatenate([med, iqr]).astype(np.float32)

    meta = {"speech_seconds": float(speech_samples / sr), "n_segments": int(len(segs))}
    return agg, meta, segs

def confidence_band(speech_seconds: float, agreement: float) -> str:
    # agreement: lower is better (e.g. variance proxy); keep v1 simple
    if speech_seconds >= 15 and agreement <= 0.08:
        return "high"
    if speech_seconds >= 6 and agreement <= 0.15:
        return "medium"
    return "low"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/audio_baseline.joblib")
    ap.add_argument("--audio", required=True)
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    pipe = bundle["pipeline"]
    sr = int(bundle.get("sr", SR_DEFAULT))

    feats, meta, segs = file_features_and_segments(args.audio, sr=sr)
    if feats is None:
        report = {
            "overall_risk": None,
            "confidence": "low",
            "error": "No usable speech detected (too short or silent).",
            "meta": meta
        }
        print(json.dumps(report, indent=2))
        return

    # Predict file-level spoof probability
    p = float(pipe.predict_proba(feats.reshape(1, -1))[0, 1])

    # Simple “agreement” proxy for v1:
    # run the classifier on segment-level aggregated vectors (median only) to estimate variance
    # (this is crude but useful)
    y, _ = librosa.load(args.audio, sr=sr, mono=True)
    seg_probs = []
    for (s, e) in segs:
        seg = y[s:e]
        sf = segment_features(seg, sr=sr)
        # for segment scoring, pad to match model input (segment_feat vs file_feat differs)
        # so we approximate by duplicating as [median, iqr=0] to match feature length
        seg_vec = np.concatenate([sf, np.zeros_like(sf)], axis=0).astype(np.float32)
        seg_probs.append(float(pipe.predict_proba(seg_vec.reshape(1, -1))[0, 1]))

    seg_var = float(np.var(seg_probs)) if len(seg_probs) >= 2 else 0.2
    conf = confidence_band(meta["speech_seconds"], seg_var)

    # Tiny “why” indicators (not perfect, but gives you something to iterate on)
    evidence = []
    if meta["speech_seconds"] < 6:
        evidence.append("Short speech duration reduces reliability.")
    if p > 0.8:
        evidence.append("Overall acoustic feature profile is strongly consistent with known spoof patterns.")
    elif p > 0.6:
        evidence.append("Multiple acoustic indicators show moderate similarity to spoof patterns.")
    else:
        evidence.append("Acoustic indicators are closer to typical human speech patterns.")

    report = {
        "overall_risk": round(p, 4),
        "confidence": conf,
        "meta": meta,
        "segment_risks": [
            {"start_s": round(s / sr, 2), "end_s": round(e / sr, 2), "risk": round(r, 4)}
            for (s, e), r in zip(segs[:10], seg_probs[:10])  # cap output
        ],
        "evidence": evidence,
        "model": {
            "feature_version": bundle.get("feature_version", "unknown"),
            "sr": sr
        }
    }

    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    main()
