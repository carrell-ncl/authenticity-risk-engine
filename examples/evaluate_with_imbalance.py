#!/usr/bin/env python
"""
Example script showing how to evaluate with imbalanced class distributions
and analyze top-k purity metrics.

This demonstrates how top-k purity is more meaningful when you have
realistic imbalanced data (e.g., only 10% of samples are fake).
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.image_det.inference.evaluate_test_set import (
    evaluate_test_set,
    recalculate_metrics_with_threshold,
    print_metrics
)


def example_balanced_vs_imbalanced():
    """Compare evaluation with balanced vs imbalanced test sets."""
    
    test_dir = r"data\image_det\Celeb-DF Preprocessed\test"
    
    print("="*80)
    print("SCENARIO 1: BALANCED TEST SET (50% fake, 50% real)")
    print("="*80)
    
    # Balanced: 50 fake, 50 real
    metrics_balanced, df_balanced = evaluate_test_set(
        test_dir=test_dir,
        samples_per_class_dict={'fake': 50, 'real': 50},
        verbose=True
    )
    
    # Calculate top-k purity with different thresholds
    print("\nRecalculating with lower threshold for better fake detection...")
    metrics_balanced_03, df_balanced_03 = recalculate_metrics_with_threshold(
        df_balanced, 
        threshold=0.3,
        top_k_percentiles=[10, 20, 30, 50]
    )
    
    print("\nTop-K Purity Analysis (Balanced, threshold=0.3):")
    for k, stats in metrics_balanced_03['top_k_purity'].items():
        print(f"\n{k.upper()} most confident predictions:")
        print(f"  Samples: {stats['n_samples']}")
        print(f"  Purity (Accuracy): {stats['purity']:.2%}")
        print(f"  Min Confidence: {stats['min_confidence']:.3f}")
        print(f"  Mean Confidence: {stats['mean_confidence']:.3f}")
        print(f"  True Distribution: {stats['true_distribution']}")
        print(f"  Predicted Distribution: {stats['predicted_distribution']}")
    
    print("\n\n" + "="*80)
    print("SCENARIO 2: IMBALANCED TEST SET (10% fake, 90% real - REALISTIC)")
    print("="*80)
    
    # Imbalanced: 10 fake, 90 real (mimics real-world where fakes are rare)
    metrics_imbalanced, df_imbalanced = evaluate_test_set(
        test_dir=test_dir,
        samples_per_class_dict={'fake': 10, 'real': 90},
        verbose=True
    )
    
    # Calculate top-k purity
    print("\nRecalculating with lower threshold for better fake detection...")
    metrics_imbalanced_03, df_imbalanced_03 = recalculate_metrics_with_threshold(
        df_imbalanced,
        threshold=0.3,
        top_k_percentiles=[10, 20, 30, 50]
    )
    
    print("\nTop-K Purity Analysis (Imbalanced, threshold=0.3):")
    for k, stats in metrics_imbalanced_03['top_k_purity'].items():
        print(f"\n{k.upper()} most confident predictions:")
        print(f"  Samples: {stats['n_samples']}")
        print(f"  Purity (Accuracy): {stats['purity']:.2%}")
        print(f"  Min Confidence: {stats['min_confidence']:.3f}")
        print(f"  Mean Confidence: {stats['mean_confidence']:.3f}")
        print(f"  True Distribution: {stats['true_distribution']}")
        print(f"  Predicted Distribution: {stats['predicted_distribution']}")
    
    print("\n\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print("""
With imbalanced data (realistic scenario):
- Top-K purity shows how well the model identifies high-confidence predictions
- Top 10% might capture most fake images if model is well-calibrated
- This metric is more meaningful than overall accuracy when classes are imbalanced
- You can set thresholds to catch fakes in top-k% most suspicious predictions

Example workflow:
1. Set threshold low (e.g., 0.3) to catch more fakes (high recall)
2. Look at top 10% most confident "fake" predictions
3. These are your priority items for human review
4. Top-k purity tells you what % of those will actually be fake
    """)


def example_top_k_usage():
    """Show how to use top-k purity in practice."""
    
    test_dir = r"data\image_det\Celeb-DF Preprocessed\test"
    
    print("="*80)
    print("PRACTICAL TOP-K PURITY WORKFLOW")
    print("="*80)
    
    # Realistic imbalanced scenario
    print("\nStep 1: Evaluate on realistic imbalanced data (5% fake, 95% real)")
    metrics, df = evaluate_test_set(
        test_dir=test_dir,
        samples_per_class_dict={'fake': 5, 'real': 95},
        verbose=True
    )
    
    print("\nStep 2: Calculate top-k purity for different confidence levels")
    metrics_adj, df_adj = recalculate_metrics_with_threshold(
        df,
        threshold=0.4,  # Lower threshold to catch more fakes
        top_k_percentiles=[5, 10, 20, 50, 100]
    )
    
    print("\nStep 3: Analyze which top-k to review")
    print("\nTop-K Analysis Results:")
    print(f"{'Top K%':<10} {'N Samples':<12} {'Purity':<10} {'Fake Count':<12} {'Real Count':<12}")
    print("-" * 60)
    
    for k, stats in metrics_adj['top_k_purity'].items():
        n_samples = stats['n_samples']
        purity = stats['purity']
        fake_pct = stats['true_distribution'].get('fake', 0)
        real_pct = stats['true_distribution'].get('real', 0)
        fake_count = int(n_samples * fake_pct)
        real_count = int(n_samples * real_pct)
        
        print(f"{k:<10} {n_samples:<12} {purity:.2%}      {fake_count:<12} {real_count:<12}")
    
    print("\nStep 4: Decision making")
    print("""
Based on top-k purity analysis:
- Review top 10% most confident predictions → Catches X fakes with Y% purity
- This focuses human review effort on highest-risk items
- Better than reviewing random samples when fakes are rare
- Adjust threshold and top-k% based on your precision/recall needs
    """)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Examples of imbalanced evaluation")
    parser.add_argument(
        "--example",
        choices=['balanced_vs_imbalanced', 'top_k_usage', 'both'],
        default='both',
        help="Which example to run"
    )
    
    args = parser.parse_args()
    
    if args.example in ['balanced_vs_imbalanced', 'both']:
        example_balanced_vs_imbalanced()
    
    if args.example in ['top_k_usage', 'both']:
        print("\n\n")
        example_top_k_usage()
