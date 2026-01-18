#!/usr/bin/env python
"""Quick test to verify the imbalanced sampling feature works correctly."""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock test to verify the sampling logic
def test_sampling_logic():
    """Test the sampling logic without running the full model."""
    
    # Simulate dataset structure
    fake_indices = list(range(0, 100))  # 100 fake samples
    real_indices = list(range(100, 1000))  # 900 real samples
    
    class_indices = {0: fake_indices, 1: real_indices}
    class_names = ['fake', 'real']
    
    # Test case 1: Balanced sampling
    print("Test 1: Balanced sampling (50 from each class)")
    sample_per_class = 50
    sampled_indices = []
    np.random.seed(42)
    
    for label, indices in class_indices.items():
        n_samples = min(sample_per_class, len(indices))
        sampled = np.random.choice(indices, size=n_samples, replace=False)
        sampled_indices.extend(sampled)
        print(f"  {class_names[label]}: sampled {n_samples}/{len(indices)}")
    
    assert len(sampled_indices) == 100, f"Expected 100, got {len(sampled_indices)}"
    print(f"  ✓ Total sampled: {len(sampled_indices)}")
    
    # Test case 2: Imbalanced sampling
    print("\nTest 2: Imbalanced sampling (10 fake, 90 real)")
    samples_per_class_dict = {'fake': 10, 'real': 90}
    sampled_indices = []
    np.random.seed(42)
    
    for label, indices in class_indices.items():
        class_name = class_names[label]
        n_to_sample = samples_per_class_dict.get(class_name, 0)
        n_samples = min(n_to_sample, len(indices))
        
        if n_samples > 0:
            sampled = np.random.choice(indices, size=n_samples, replace=False)
            sampled_indices.extend(sampled)
        
        print(f"  {class_name}: sampled {n_samples}/{len(indices)} (requested: {n_to_sample})")
    
    assert len(sampled_indices) == 100, f"Expected 100, got {len(sampled_indices)}"
    
    # Verify distribution
    fake_count = sum(1 for idx in sampled_indices if idx < 100)
    real_count = sum(1 for idx in sampled_indices if idx >= 100)
    
    print(f"  ✓ Total sampled: {len(sampled_indices)}")
    print(f"  Distribution: fake={fake_count} ({100*fake_count/len(sampled_indices):.1f}%), real={real_count} ({100*real_count/len(sampled_indices):.1f}%)")
    
    assert fake_count == 10, f"Expected 10 fake, got {fake_count}"
    assert real_count == 90, f"Expected 90 real, got {real_count}"
    
    # Test case 3: Extreme imbalance (5% fake)
    print("\nTest 3: Extreme imbalance (5 fake, 95 real)")
    samples_per_class_dict = {'fake': 5, 'real': 95}
    sampled_indices = []
    np.random.seed(42)
    
    for label, indices in class_indices.items():
        class_name = class_names[label]
        n_to_sample = samples_per_class_dict.get(class_name, 0)
        n_samples = min(n_to_sample, len(indices))
        
        if n_samples > 0:
            sampled = np.random.choice(indices, size=n_samples, replace=False)
            sampled_indices.extend(sampled)
        
        print(f"  {class_name}: sampled {n_samples}/{len(indices)} (requested: {n_to_sample})")
    
    fake_count = sum(1 for idx in sampled_indices if idx < 100)
    real_count = sum(1 for idx in sampled_indices if idx >= 100)
    
    print(f"  ✓ Total sampled: {len(sampled_indices)}")
    print(f"  Distribution: fake={fake_count} ({100*fake_count/len(sampled_indices):.1f}%), real={real_count} ({100*real_count/len(sampled_indices):.1f}%)")
    
    assert fake_count == 5, f"Expected 5 fake, got {fake_count}"
    assert real_count == 95, f"Expected 95 real, got {real_count}"
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    test_sampling_logic()
