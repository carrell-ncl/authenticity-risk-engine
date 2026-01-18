# Production Model Strategy for Insurance Fraud Detection

## Current Problem
- Your model trained on one dataset, tested on another → 50% accuracy
- Celebrity deepfakes ≠ insurance fraud images
- Need generalization to real-world manipulations

## FIRST STEP: Data Strategy

### 1. What Insurance Fraud Looks Like
Insurance manipulation types:
- **Photoshopped damage** (cars, property)
- **Face swaps** (identity fraud)
- **Document tampering** (receipts, invoices)
- **AI-generated fake scenes** (staged accidents)
- **Image splicing** (combining multiple images)
- **Copy-move forgery** (duplicating parts of image)

### 2. Recommended Training Datasets

#### Must-Have Datasets (Download Priority):

**A. FaceForensics++** ⭐ HIGHEST PRIORITY
- Download: https://github.com/ondyari/FaceForensics
- Why: Multiple manipulation methods (not just face swap)
- Contains: 4 manipulation techniques + pristine originals
- Size: ~500GB (get compressed version ~10GB)
- Methods: Deepfakes, Face2Face, FaceSwap, NeuralTextures

**B. Celeb-DF v2** (You already have this)
- Keep for face-focused manipulations
- Good for identity fraud detection

**C. CASIA v2** ⭐ IMPORTANT for insurance
- Download: https://github.com/namtpham/casia2groundtruth
- Why: Copy-move, splicing forgery (common in insurance fraud!)
- Size: ~7GB
- Contains: Real-world style manipulations

**D. COVERAGE Dataset** (If available)
- Insurance-specific if you can get it
- Real fraud examples from insurance companies

### 3. Data Collection Plan

#### Week 1: Download & Organize
```bash
data/
  training/
    real/
      faceforensics_real_001.jpg
      casia_real_001.jpg
      celeb_real_001.jpg
    fake/
      faceforensics_deepfake_001.jpg
      faceforensics_face2face_001.jpg
      casia_splice_001.jpg
      casia_copymove_001.jpg
      celeb_fake_001.jpg
```

#### Week 2: Data Preparation
- Balance classes (equal real/fake)
- Remove corrupted images
- Standardize format (RGB, proper dimensions)
- Create train/val/test splits (70/15/15)

### 4. Data Quality Checks
```python
# Check dataset balance
# Verify image quality
# Ensure no data leakage
# Test on held-out data
```

## SECOND STEP: Training Strategy

### Training Configuration
```bash
python src/image_det/train/train_deepfake.py \
    --data_dir "data/training" \
    --out_dir "models/image_det/production" \
    --epochs 30 \
    --batch_size 32 \
    --model_name efficientnet_b3 \
    --lr 5e-5 \
    --patience 7 \
    --val_split 0.15
```

### Model Selection
- Start: EfficientNet-B3 (good balance)
- If accuracy insufficient: EfficientNet-B4 or B5
- Consider ensemble for critical decisions

## THIRD STEP: Validation

### Test on Multiple Scenarios
1. Face manipulations (identity fraud)
2. Scene manipulations (staged damage)
3. Document alterations
4. Real insurance claim images (if available)

### Success Metrics for Insurance
- **Accuracy > 95%** (insurance requires high confidence)
- **False Negative Rate < 2%** (don't miss fraud!)
- **False Positive Rate < 5%** (don't flag legitimate claims)
- **Generalization**: Similar accuracy across all test sets

## ACTION ITEMS - START TODAY

### Immediate (Today):
1. ✅ Download FaceForensics++ (compressed version)
2. ✅ Download CASIA v2
3. ✅ Create organized folder structure

### This Week:
1. ✅ Prepare combined training dataset
2. ✅ Verify data quality and balance
3. ✅ Create proper train/val/test splits
4. ✅ Train first production model

### Next Week:
1. ✅ Test on multiple datasets
2. ✅ Measure cross-dataset performance
3. ✅ Fine-tune based on results
4. ✅ Deploy best model to API

## Data Download Commands

```bash
# FaceForensics++ (Request access first)
# Fill form: https://github.com/ondyari/FaceForensics

# CASIA v2
wget "download_link" -O casia2.zip
unzip casia2.zip -d data/casia2/

# Alternative: Kaggle datasets
kaggle datasets download -d user/faceforensics
```

## Expected Timeline
- **Week 1**: Data collection and preparation
- **Week 2**: Initial training (3-5 days)
- **Week 3**: Testing and validation
- **Week 4**: Production deployment

## Success Definition
✅ Model accuracy > 95% on held-out test sets
✅ Generalizes across different manipulation types
✅ Fast inference (< 200ms per image)
✅ Deployed API with monitoring
✅ Frontend for testing
