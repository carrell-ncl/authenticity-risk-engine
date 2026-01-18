# Imbalanced Dataset Evaluation Guide

## Overview

The updated `evaluate_test_set.py` script now supports custom sampling per class, allowing you to simulate real-world scenarios where fake samples are rare (e.g., only 5-10% of data).

This makes **top-k purity** metrics much more meaningful for production scenarios.

## What's New

### 1. Custom Samples Per Class

You can now specify different numbers of samples for each class:

```python
from src.image_det.inference.evaluate_test_set import evaluate_test_set

# Imbalanced: 10 fake, 90 real (10% fake - realistic scenario)
metrics, df = evaluate_test_set(
    test_dir="data/image_det/Celeb-DF Preprocessed/test",
    samples_per_class_dict={'fake': 10, 'real': 90}
)
```

### 2. CLI Support

```bash
# Balanced sampling (backward compatible)
python evaluate_test_set.py --sample_per_class 50

# Imbalanced sampling - 10% fake
python evaluate_test_set.py --samples_fake 10 --samples_real 90

# More realistic - 5% fake
python evaluate_test_set.py --samples_fake 5 --samples_real 95
```

## Why This Matters for Top-K Purity

### Traditional Balanced Evaluation
- 50% fake, 50% real
- Accuracy is meaningful, but not realistic
- Top-k purity doesn't tell you much

### Realistic Imbalanced Evaluation  
- 5-10% fake, 90-95% real
- **Top-k purity becomes critical** - shows how well you can prioritize review
- Example: "Top 10% most suspicious predictions have 80% purity" means if you review just the top 10%, you'll be right 80% of the time

## Complete Workflow Example

### Step 1: Evaluate with Realistic Distribution

```python
# Simulate production: only 5% of content is fake
metrics, df = evaluate_test_set(
    test_dir="data/image_det/Celeb-DF Preprocessed/test",
    samples_per_class_dict={'fake': 5, 'real': 95},
    return_dataframe=True
)
```

### Step 2: Adjust Threshold for Better Fake Detection

```python
from src.image_det.inference.evaluate_test_set import recalculate_metrics_with_threshold

# Lower threshold to catch more fakes (higher sensitivity)
metrics_adjusted, df_adjusted = recalculate_metrics_with_threshold(
    df,
    threshold=0.3,  # Default is 0.5
    top_k_percentiles=[5, 10, 20, 30]
)
```

### Step 3: Analyze Top-K Purity

```python
# Check purity at different levels
for k, stats in metrics_adjusted['top_k_purity'].items():
    print(f"{k}: {stats['n_samples']} samples, {stats['purity']:.1%} purity")
    print(f"  True distribution: {stats['true_distribution']}")
```

**Example Output:**
```
top_5: 5 samples, 100.0% purity
  True distribution: {'fake': 1.0}
  
top_10: 10 samples, 90.0% purity
  True distribution: {'fake': 0.8, 'real': 0.2}
  
top_20: 20 samples, 75.0% purity
  True distribution: {'fake': 0.5, 'real': 0.5}
```

### Step 4: Make Decisions

- **If top 10% has 90% purity:** Only review those 10% to catch most fakes efficiently
- **If top 20% has 75% purity:** Review top 20% to catch more fakes (but with more false positives)
- Adjust threshold and review budget based on your precision/recall requirements

## Real-World Use Case

**Scenario:** Content moderation platform with 1 million images/day

**Without imbalanced testing:**
- Evaluate on 50/50 split
- Get 95% accuracy (seems great!)
- Deploy to production
- Actually only 1% of real content is fake
- Model flags 100k images as suspicious (10% false positive rate)
- Human reviewers overwhelmed

**With imbalanced testing:**
- Evaluate on 1% fake / 99% real split
- Adjust threshold to get 80% recall on fakes
- Check top-10% purity: 15% (means 1.5% of top 10% are actually fake)
- Decision: Review top 10% most suspicious → 100k images to review, catch 15k fakes
- OR adjust threshold to reduce review volume while maintaining acceptable recall

## Key Metrics Explained

### Overall Accuracy
- Less meaningful with imbalanced data
- Can be high even if model misses all fakes

### Precision/Recall (per class)
- Better than accuracy for imbalanced data
- Shows performance on each class separately

### Top-K Purity ⭐ (Most Important)
- **What it means:** Of the top K% most confident predictions, what % are correct?
- **Why it matters:** Tells you if you can trust your model's confidence scores
- **How to use:** If top 10% has 80%+ purity, focus your review efforts there

### Confusion Matrix
- Shows specific error types
- False positives (real marked as fake) vs False negatives (fake marked as real)

## Run the Examples

```bash
# See balanced vs imbalanced comparison
python examples/evaluate_with_imbalance.py --example balanced_vs_imbalanced

# See practical top-k usage
python examples/evaluate_with_imbalance.py --example top_k_usage

# Run both
python examples/evaluate_with_imbalance.py --example both
```

## Tips

1. **Start with realistic class distributions** from your production data
2. **Experiment with thresholds** - lower catches more fakes but increases false positives
3. **Focus on top-k purity** for actionable insights
4. **Set decision rules** based on your business constraints (review budget, false positive tolerance)
5. **Track metrics over time** as your data distribution changes

## API Reference

### evaluate_test_set()

```python
def evaluate_test_set(
    test_dir: str,
    checkpoint_path: Optional[str] = None,
    samples_per_class_dict: Optional[Dict[str, int]] = None,
    seed: int = 42,
    # ... other params
) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
```

**Key Parameters:**
- `samples_per_class_dict`: Dict like `{'fake': 10, 'real': 90}` for custom sampling
- `sample_per_class`: Integer for balanced sampling (backward compatible)
- `seed`: Random seed for reproducibility

### recalculate_metrics_with_threshold()

```python
def recalculate_metrics_with_threshold(
    df_results: pd.DataFrame,
    threshold: float = 0.5,
    top_k_percentiles: Optional[List[int]] = None
) -> Tuple[Dict[str, Any], pd.DataFrame]:
```

**Key Parameters:**
- `threshold`: Classification threshold (0-1), lower = more sensitive to fakes
- `top_k_percentiles`: List like `[10, 20, 30]` for top-k analysis

## Additional Resources

- [evaluate_test_set.py](../src/image_det/inference/evaluate_test_set.py) - Main evaluation script
- [evaluate_with_imbalance.py](../examples/evaluate_with_imbalance.py) - Example usage
- [HANDLING_IMBALANCE.md](./HANDLING_IMBALANCE.md) - General imbalance handling strategies
