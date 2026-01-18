"""
Organize CASIA v2 Dataset into Train/Val/Test Splits

This script takes the CASIA v2 dataset and creates proper train/val/test splits
with the correct folder structure for training.

Usage:
    python organize_casia_dataset.py
"""

import os
import shutil
from pathlib import Path
import random
from tqdm import tqdm

# Configuration
CASIA_ROOT = Path('data/image_det/CASIA2')
OUTPUT_ROOT = Path('data/image_det/CASIA2_organized')

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Balance dataset by undersampling majority class
BALANCE_DATASET = True  # Set to True to balance real/fake classes

# Set seed for reproducibility
random.seed(42)


def organize_dataset():
    """Organize CASIA v2 into train/val/test splits"""
    
    print("="*60)
    print("CASIA v2 Dataset Organization")
    print("="*60)
    
    # Check if CASIA dataset exists
    au_dir = CASIA_ROOT / 'real'
    tp_dir = CASIA_ROOT / 'fake'
    
    if not au_dir.exists() or not tp_dir.exists():
        print(f"❌ CASIA dataset not found at {CASIA_ROOT}")
        print(f"Expected folders: real/ and fake/")
        return
    
    # Get all image files
    print("\n📁 Scanning dataset...")
    au_images = list(au_dir.glob('*.*'))
    tp_images = list(tp_dir.glob('*.*'))
    
    print(f"Found {len(au_images)} authentic (real) images")
    print(f"Found {len(tp_images)} tampered (fake) images")
    
    # Balance dataset if requested
    if BALANCE_DATASET:
        min_count = min(len(au_images), len(tp_images))
        print(f"\n⚖️  Balancing dataset to {min_count} images per class...")
        print(f"   Undersampling real images: {len(au_images)} → {min_count}")
        print(f"   Fake images: {len(tp_images)} (unchanged)" if len(tp_images) <= len(au_images) else f"   Undersampling fake images: {len(tp_images)} → {min_count}")
        
        au_images = random.sample(au_images, min_count)
        tp_images = random.sample(tp_images, min_count)
        
        print(f"✅ Balanced: {len(au_images)} real + {len(tp_images)} fake = {len(au_images) + len(tp_images)} total")
    else:
        print(f"\n⚠️  Dataset is imbalanced (ratio {len(au_images)/max(len(tp_images),1):.2f}:1)")
        print(f"   Class weights will be used during training to handle imbalance")
    
    # Shuffle for random splits
    random.shuffle(au_images)
    random.shuffle(tp_images)
    
    # Calculate split indices
    def get_splits(images):
        n = len(images)
        train_end = int(n * TRAIN_RATIO)
        val_end = train_end + int(n * VAL_RATIO)
        
        return {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:]
        }
    
    au_splits = get_splits(au_images)
    tp_splits = get_splits(tp_images)
    
    print(f"\n📊 Split sizes:")
    print(f"  Train: {len(au_splits['train'])} real + {len(tp_splits['train'])} fake = {len(au_splits['train']) + len(tp_splits['train'])}")
    print(f"  Val:   {len(au_splits['val'])} real + {len(tp_splits['val'])} fake = {len(au_splits['val']) + len(tp_splits['val'])}")
    print(f"  Test:  {len(au_splits['test'])} real + {len(tp_splits['test'])} fake = {len(au_splits['test']) + len(tp_splits['test'])}")
    
    # Check balance in each split
    for split_name in ['train', 'val', 'test']:
        real_count = len(au_splits[split_name])
        fake_count = len(tp_splits[split_name])
        ratio = real_count / max(fake_count, 1)
        balance_status = "✅ Balanced" if abs(ratio - 1.0) < 0.1 else f"⚠️  Imbalanced (ratio {ratio:.2f}:1)"
        print(f"    {split_name.capitalize()}: {balance_status}")
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (OUTPUT_ROOT / split / 'real').mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / split / 'fake').mkdir(parents=True, exist_ok=True)
    
    # Copy files to organized structure
    print("\n📦 Copying files...")
    
    for split in ['train', 'val', 'test']:
        print(f"\n  {split.upper()}:")
        
        # Copy authentic (real) images
        for img in tqdm(au_splits[split], desc=f"    Copying real images"):
            dest = OUTPUT_ROOT / split / 'real' / img.name
            shutil.copy2(img, dest)
        
        # Copy tampered (fake) images
        for img in tqdm(tp_splits[split], desc=f"    Copying fake images"):
            dest = OUTPUT_ROOT / split / 'fake' / img.name
            shutil.copy2(img, dest)
    
    print("\n" + "="*60)
    print("✅ Dataset organized successfully!")
    print("="*60)
    print(f"\nOrganized dataset location: {OUTPUT_ROOT}")
    print("\nFolder structure:")
    print("  CASIA2_organized/")
    print("    train/")
    print("      real/")
    print("      fake/")
    print("    val/")
    print("      real/")
    print("      fake/")
    print("    test/")
    print("      real/")
    print("      fake/")
    
    # Print training command
    print("\n🚀 Ready to train! Use this command:")
    print(f"\npython src/image_det/train/train_deepfake.py \\")
    print(f"    --data_dir \"{OUTPUT_ROOT / 'train'}\" \\")
    print(f"    --out_dir \"models/image_det/runs_casia\" \\")
    print(f"    --epochs 20 \\")
    print(f"    --batch_size 32 \\")
    print(f"    --model_name efficientnet_b0")
    
    print("\n📊 Or test on it:")
    print(f"\nfrom src.image_det.inference.evaluate_test_set import evaluate_test_set")
    print(f"metrics, df = evaluate_test_set(test_dir=r\"{OUTPUT_ROOT / 'test'}\")")


if __name__ == "__main__":
    organize_dataset()
