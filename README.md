# authenticity-risk-engine
This repository implements an audio authenticity risk scoring system designed to detect synthetic / spoofed speech and prioritise the most suspicious samples for review.

The system is not a binary deepfake detector.
It is a risk-ranking and calibration pipeline suitable for fraud investigation and regulated environments (e.g. insurance, compliance, SIU workflows).

A system that prioritises suspicious audio for investigation, rather than making hard decisions.

A configurable authenticity risk engine that plugs into investigation workflows and reduces review cost.

# Datasets
https://datashare.ed.ac.uk/handle/10283/3336
https://zenodo.org/records/4835108
https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset?resource=download


## High-level design

The system is composed of two stages:

```
Audio file
   ↓
CNN spoof detector (spectrogram-based)
   ↓
Segment-level scoring
   ↓
Feature aggregation
   ↓
Logistic regression aggregator
   ↓
Calibrated authenticity risk score
```

## Scoring policy:

if domain == ASVspoof-like:
    score = cnn_prob
elif domain == RealWorld-like:
    score = rof_calibrator_prob
else:
    if uncertainty > threshold:
        score = cnn_prob
        flag = "domain_shift"
    else:
        score = rof_calibrator_prob

### Design principles
- High recall for synthetic audio
- Controlled false positives via calibration
- Domain-specific behaviour without retraining the CNN
- Interpretable, auditable outputs

---

## Stage 1 — CNN spoof detector

### Model
- Lightweight 2D CNN (`SmallResNet`)
- Input: log-mel spectrograms
- Output: probability of synthetic speech

### Training characteristics
- Trained on ASVspoof-style datasets
- Uses fixed-length clips (e.g. 4 seconds)
- Learns **local spectral artefacts** associated with speech synthesis

### Behaviour
- Strong cross-dataset generalisation on fake audio
- Very high fake recall
- Conservative on some “clean real” domains (expected behaviour)

The CNN is treated as a **frozen signal generator** in downstream stages.

---

## Stage 2 — Aggregator (Logistic Regression)

### Purpose
The aggregator does **not** operate on raw audio features.

It learns how to **interpret CNN behaviour** and calibrate risk for a specific data domain, reducing false positives while preserving fake recall.

### Input features
For each audio file, the following features are extracted:

| Feature | Description |
|------|------------|
| `cnn_median` | Median CNN score across segments |
| `cnn_max` | Maximum CNN score (worst-case evidence) |
| `cnn_var` | Variance of CNN scores (stability across time) |
| `total_seconds` | Duration of the audio file |
| `silence_ratio` | Fraction of low-amplitude samples |

These features capture:
- strength of evidence
- agreement across segments
- amount of speech information available

### Model
- Logistic regression
- Standardised features
- Class-balanced loss
- Used strictly as a **combiner and probability calibrator**

---

## Evaluation philosophy

### What is *not* optimised
- Raw overall accuracy
- Single fixed-threshold classification
- Standalone handcrafted-feature detection

### What *is* optimised
- Ranking quality (AUC)
- False-positive control
- Review efficiency
- Domain-specific calibration

---

## Metrics reported

### AUC (ROC)
Measures ranking quality (fake vs real).

- AUC ≈ 0.92 indicates excellent separability
- This is the primary health metric

---

### Threshold metrics
Reported at several operating points (e.g. 0.3, 0.5, 0.7, 0.85):

- real accuracy  
- fake accuracy  
- precision  
- confusion matrix  

These allow tuning aggressiveness based on operational needs.

---

### Top-K review metrics (primary)
Answers the question:

> “If only the top X% most suspicious files are reviewed, how effective is that review?”

Example outcomes:
- Top 10% reviewed
- ~95% of reviewed files are fake
- ~18–20% of all fakes captured

This reflects **real investigative workflows**.

---

## Why two stages are required

| Approach | Outcome |
|------|------|
| Handcrafted LR only | Fails under domain shift |
| CNN only | High recall, excessive false positives |
| **CNN + aggregator** | High recall *and* controlled false positives |

The aggregator adapts CNN behaviour **without retraining the CNN**.

---

## Dataset usage

### Supported evaluation setups
- ASVspoof-style datasets (FLAC)
- Real-vs-Fake datasets (WAV)
- Mixed codecs and sample rates

Audio format does not matter; all files are decoded to waveforms.

### Important rule
**Each dataset/domain should have its own aggregator.**  
CNN weights remain shared across domains.

---

## Performance summary (example)

- CNN fake accuracy (cross-dataset): ~99%
- CNN real accuracy (unadjusted): low on some domains
- Aggregated system:
  - AUC ≈ 0.92
  - Real accuracy improved to 80–95% (threshold-dependent)
  - High-purity review queues (≈95% fake in top 10%)

This behaviour is **intentional** and **desirable** for risk scoring.

---

## Running the aggregator

```bash
python train_aggregator_lr.py \
  --cnn_model models/audio_cnn_mel.pt \
  --data_dir data/eval_dataset \
  --out models/agg_lr.joblib \
  --clip_seconds 4.0 \
  --n_segments 6 \
  --device cuda
```

### Options
- `--device cuda` enables GPU acceleration  
- `--n_segments` controls stability vs speed  
- `--clip_seconds` should match CNN training  

---

## Output artefacts

- Trained aggregator model (`.joblib`)
- JSON report including:
  - AUC
  - threshold metrics
  - top-K review metrics
- Feature names and configuration metadata

---

## Intended use

This system is designed to:
- prioritise suspicious audio
- support investigator decision-making
- operate conservatively under uncertainty
- adapt across domains without retraining core models

It is **not** intended to make irreversible automated decisions.

---

## Current limitations

- Audio-only (no video or images yet)
- Domain selection is manual
- Silence/SNR features are heuristic
- Batch inference can be further optimised

These are deliberate design trade-offs at this stage.

---

## Roadmap (high level)

- Batched GPU inference  
- Additional confidence features (SNR, entropy)  
- Unified scoring API  
- Video and image modalities  
- Multi-modal aggregation  

---

## Summary

This repository implements a **production-oriented audio authenticity risk engine** that:

- generalises across datasets  
- prioritises review efficiency  
- avoids brittle handcrafted detection  
- remains interpretable and controllable  

It reflects how real fraud and investigation systems are built — not how benchmark classifiers are optimised.

## Architecture

┌───────────────────────────────┐
│           Audio File           │
│      (.wav, .flac, etc.)       │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│      Audio Preprocessing       │
│  - decode waveform             │
│  - mono conversion             │
│  - resample to target SR       │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│        Segment Sampling        │
│  - fixed-length clips          │
│  - sampled across duration     │
│  - e.g. 6 × 4-second segments  │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│     CNN Spoof Signal Model     │
│          (SmallResNet)         │
│                               │
│  For each segment:             │
│   - log-mel spectrogram        │
│   - CNN forward pass           │
│   - segment spoof prob ∈ [0,1] │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│     Segment Score Set          │
│   [p₁, p₂, p₃, ..., pₙ]        │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│     Feature Aggregation        │
│                               │
│  Derived signals:              │
│   - cnn_median                 │
│   - cnn_max                    │
│   - cnn_var                    │
│   - total_seconds              │
│   - silence_ratio              │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────────────────────┐
│      Domain Calibration Layer (LR models)      │
│                                               │
│  Calibrator A: ASVspoof / lab-style            │
│  Calibrator B: Real-world / “wild audio”       │
│                                               │
│  Outputs:                                      │
│   - pA = P(fake | features, calibrator A)      │
│   - pB = P(fake | features, calibrator B)      │
└───────────────┬───────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────┐
│   Uncertainty + Routing / Safety Policy        │
│                                               │
│  Disagreement / domain shift:                  │
│   - u = |pA - pB|                              │
│                                               │
│  If domain known:                              │
│   - ASV-like → use pA or CNN-only              │
│   - Real-world → use pB                        │
│                                               │
│  If domain unknown:                            │
│   - if u > threshold → fallback to CNN score   │
│   - else → use selected calibrator score       │
│     (e.g. calibrator closer to CNN signal)     │
└───────────────┬───────────────────────────────┘
                │
                ▼
┌───────────────────────────────┐
│     Risk-Based Decision Layer  │
│                               │
│  Examples:                     │
│   - thresholding               │
│   - top-K review ranking       │
│   - tiering (High/Med/Low)     │
│   - uncertainty flagging       │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│     Investigator / System      │
│     Output                     │
│                               │
│  - risk score                  │
│  - uncertainty / domain flag   │
│  - explanation signals         │
│  - audit-friendly metadata     │
└───────────────────────────────┘


