# How to Download FaceForensics++ Dataset

## Method 1: Official Download Script (Recommended)

### Step 1: Get Access
1. Visit: https://github.com/ondyari/FaceForensics
2. Scroll to "Access" section
3. Fill out the Google Form: https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform
4. You'll receive download credentials via email (usually within 1-2 days)

### Step 2: Install Requirements
```bash
pip install requests tqdm
```

### Step 3: Download Using Official Script

Once you receive credentials, use their download script:

```bash
# Clone the repo
git clone https://github.com/ondyari/FaceForensics.git
cd FaceForensics

# Run download script
python download-FaceForensics.py \
    --server EU \
    --dataset FaceForensics++ \
    --compression c23 \
    --type images \
    --num_videos all

# Options:
# --server: EU or US
# --compression: c23 (compressed, ~10GB) or c0 (raw, ~500GB)
# --type: images or videos or masks
# --dataset: FaceForensics++
```

### Step 4: Organize Downloaded Data
```bash
# After download, organize like this:
mv FaceForensics++/original_sequences/youtube/c23/images "data/image_det/FaceForensics++/original"
mv FaceForensics++/manipulated_sequences/Deepfakes/c23/images "data/image_det/FaceForensics++/Deepfakes"
mv FaceForensics++/manipulated_sequences/Face2Face/c23/images "data/image_det/FaceForensics++/Face2Face"
mv FaceForensics++/manipulated_sequences/FaceSwap/c23/images "data/image_det/FaceForensics++/FaceSwap"
mv FaceForensics++/manipulated_sequences/NeuralTextures/c23/images "data/image_det/FaceForensics++/NeuralTextures"
```

## Method 2: Alternative Sources (If Official is Slow)

### Kaggle (Sometimes Available)
```bash
# Install Kaggle CLI
pip install kaggle

# Search for FaceForensics
kaggle datasets list -s faceforensics

# Download if available
kaggle datasets download -d [dataset-name]
```

### Academic Torrents (Sometimes Available)
- Visit: http://academictorrents.com/
- Search: "FaceForensics"
- Use torrent client to download

## Method 3: Quick Start with Subset

If you want to start immediately with a smaller subset:

```python
# download_faceforensics_subset.py
import requests
from pathlib import Path
import zipfile
from tqdm import tqdm

def download_file(url, destination):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        desc=destination.name,
        total=total_size,
        unit='iB',
        unit_scale=True
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            pbar.update(size)

# Note: You'll need credentials from the form
# This is just the structure - actual URLs require authentication
```

## What You'll Get

### Directory Structure:
```
FaceForensics++/
├── original/           # Real videos/images (~7,000 videos)
├── Deepfakes/         # Deepfake manipulations
├── Face2Face/         # Face2Face manipulations
├── FaceSwap/          # FaceSwap manipulations
└── NeuralTextures/    # NeuralTextures manipulations
```

### Dataset Size:
- **c23 (compressed)**: ~10-15 GB (recommended for training)
- **c40 (medium)**: ~50-100 GB
- **c0 (raw)**: ~500 GB

## Recommended Download Command

```bash
# For image training (what you need):
python download-FaceForensics.py \
    --server EU \
    --dataset FaceForensics++ \
    --compression c23 \
    --type images \
    --num_videos all \
    --output_path "c:/Users/Steve/Desktop/dev/authenticity-risk-engine/data/image_det/FaceForensics++"
```

## Troubleshooting

### "Access Denied"
- Make sure you filled out the Google Form
- Check your email for credentials
- Wait 1-2 business days for approval

### "Download Too Slow"
- Try different server (EU vs US)
- Use `--num_videos 100` to download subset first
- Download during off-peak hours

### "Not Enough Space"
- Use c23 compression (smallest)
- Download only specific manipulation types:
  ```bash
  # Just Deepfakes and original
  python download-FaceForensics.py ... --method Deepfakes
  ```

## Alternative: Use What You Have

While waiting for FaceForensics++, you can:

1. **Train on Celeb-DF only** (you already have it)
2. **Download CASIA v2** (easier to get, no registration needed)
3. **Use data augmentation** heavily to improve generalization

## After Download

Once downloaded, run the data preparation notebook:
```bash
jupyter notebook notebooks/image_det/prepare_production_data.ipynb
```

It will combine FaceForensics++ with your existing Celeb-DF data for better training.

## Estimated Timeline

- **Form submission**: 2 minutes
- **Approval wait**: 1-2 days
- **Download time**: 2-6 hours (depends on connection and compression level)
- **Extraction**: 30 minutes
- **Organization**: 15 minutes

**Total**: Start now, ready to train in 2-3 days!
