# Handling Class Imbalance in CASIA Training

## Problem
CASIA dataset is imbalanced: ~12,000 real vs ~3,500 fake images (3.4:1 ratio)

## Solutions Implemented

### Option 1: Balance Dataset Before Training (Recommended)
**Undersample majority class to match minority class**

```python
# In organize_casia_dataset.py, set:
BALANCE_DATASET = True
```

Then run:
```bash
python src/data/organize_casia_dataset.py
```

Result: ~3,500 real + ~3,500 fake = 7,000 balanced images

✅ **Pros:** Simple, clean, best for training
❌ **Cons:** Discards data from majority class

### Option 2: Use Class Weights in Loss (Default, Automatic)
**Train on full imbalanced dataset with weighted loss**

Training script automatically calculates class weights:
```bash
python src/image_det/train/train_deepfake.py \
    --data_dir "data/image_det/CASIA2_organized/train" \
    --epochs 20 \
    --batch_size 32
```

The model will use `pos_weight = 12000/3500 = 3.43` in BCEWithLogitsLoss
This penalizes fake class errors more heavily.

✅ **Pros:** Uses all data
❌ **Cons:** May still bias toward majority class

### Option 3: Weighted Random Sampling
**Oversample minority class during training**

```bash
python src/image_det/train/train_deepfake.py \
    --data_dir "data/image_det/CASIA2_organized/train" \
    --epochs 20 \
    --batch_size 32 \
    --use_weighted_sampler
```

This ensures each batch has ~50/50 real/fake by sampling minority class more frequently.

✅ **Pros:** Balanced batches, uses all data
❌ **Cons:** Minority class images seen more often (may overfit)

### Option 4: Combine Methods
**Best approach for production**

1. Balance dataset (Option 1)
2. Use class weights (automatically enabled)
3. Monitor both classes separately during validation

```bash
# First organize with balancing
python src/data/organize_casia_dataset.py

# Then train
python src/image_det/train/train_deepfake.py \
    --data_dir "data/image_det/CASIA2_organized/train" \
    --epochs 25 \
    --batch_size 32 \
    --model_name efficientnet_b3 \
    --lr 5e-5 \
    --patience 5
```

## Training Command Examples

### Balanced Dataset (Recommended)
```bash
# Step 1: Balance data
python src/data/organize_casia_dataset.py

# Step 2: Train
python src/image_det/train/train_deepfake.py \
    --data_dir "data/image_det/CASIA2_organized/train" \
    --out_dir "models/image_det/runs_casia_balanced" \
    --epochs 25 \
    --batch_size 32 \
    --model_name efficientnet_b0 \
    --patience 5
```

### Imbalanced Dataset with Weighted Sampler
```bash
# In organize_casia_dataset.py, set: BALANCE_DATASET = False
python src/data/organize_casia_dataset.py

# Train with weighted sampling
python src/image_det/train/train_deepfake.py \
    --data_dir "data/image_det/CASIA2_organized/train" \
    --out_dir "models/image_det/runs_casia_weighted" \
    --epochs 25 \
    --batch_size 32 \
    --use_weighted_sampler \
    --patience 5
```

## What Gets Printed During Training

You'll see:
```
Class distribution in training:
  Label 0 (real): 8400 samples
  Label 1 (fake): 2450 samples
  Imbalance ratio: 3.43:1
  Using class weight (pos_weight): 3.429
```

OR if balanced:
```
Class distribution in training:
  Label 0 (real): 2450 samples
  Label 1 (fake): 2450 samples
  Imbalance ratio: 1.00:1
  Using class weight (pos_weight): 1.000
```

## Validation Metrics to Watch

During training, monitor BOTH classes:
```
Epoch 05/25 | lr=2.38e-04 | train loss=0.0124 acc=0.9956 | val loss=0.0126 acc=0.9952
```

After training, evaluate per-class performance:
```python
from src.image_det.inference.evaluate_test_set import evaluate_test_set

metrics, df = evaluate_test_set(
    test_dir=r"data/image_det/CASIA2_organized/test",
    verbose=True
)

# Check per-class metrics
print(metrics['class_metrics'])
```

Look for:
- Fake precision/recall (catching fraud is critical!)
- Real precision/recall (don't flag legitimate images)
- ROC-AUC score

## Recommendation for Insurance Production

**Use balanced dataset (Option 1) because:**
1. ✅ Simplest approach
2. ✅ Best generalization
3. ✅ Equal importance to both classes
4. ✅ 7,000 images still plenty for training
5. ✅ Validation/test also balanced for accurate metrics

Then **combine with other datasets** (Celeb-DF, FaceForensics++) for diversity!

```bash
# Step 1: Organize CASIA (balanced)
python src/data/organize_casia_dataset.py

# Step 2: Add to combined dataset using prepare_production_data.ipynb
jupyter notebook notebooks/image_det/prepare_production_data.ipynb

# Step 3: Train on combined balanced dataset
python src/image_det/train/train_deepfake.py \
    --data_dir "data/image_det/production_training" \
    --epochs 30 \
    --batch_size 32 \
    --model_name efficientnet_b3
```
