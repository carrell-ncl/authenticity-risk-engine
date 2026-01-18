# Quick Reference: Imbalanced Evaluation Update

## Summary of Changes

Updated [evaluate_test_set.py](../src/image_det/inference/evaluate_test_set.py) to support custom sampling per class, enabling realistic imbalanced test scenarios.

## What Changed

### New Parameters

**`samples_per_class_dict`** (Dict[str, int], optional)
- Specify different sample counts per class
- Example: `{'fake': 10, 'real': 90}` for 10% fake distribution
- Enables simulation of real-world imbalanced scenarios

**CLI Arguments**
- `--samples_fake N`: Number of fake samples to use
- `--samples_real N`: Number of real samples to use
- Works alongside existing `--sample_per_class` (for balanced sampling)

## Quick Start

### Python/Notebook Usage

```python
from src.image_det.inference.evaluate_test_set import evaluate_test_set

# Imbalanced: 10% fake (realistic)
metrics, df = evaluate_test_set(
    test_dir="data/image_det/Celeb-DF Preprocessed/test",
    samples_per_class_dict={'fake': 10, 'real': 90}
)
```

### Command Line Usage

```bash
# Imbalanced: 10% fake
python src/image_det/inference/evaluate_test_set.py \
    --samples_fake 10 \
    --samples_real 90

# Balanced (backward compatible)
python src/image_det/inference/evaluate_test_set.py \
    --sample_per_class 50
```

## Why This Matters

### Before
- Could only sample equal numbers from each class
- Metrics optimized for balanced data
- Top-k purity not very meaningful

### After  
- Can simulate realistic class distributions (e.g., 5% fake)
- **Top-k purity becomes critical metric**
- Example: "Top 10% most suspicious predictions have 85% purity"
  - Tells you if reviewing top 10% is worthwhile
  - Guides threshold and review budget decisions

## Key Benefit: Better Top-K Purity Analysis

When data is imbalanced (like production), top-k purity shows:
1. **Can you trust model confidence?** High purity in top-k% means yes
2. **Where to focus review effort?** Review the top-k% with highest purity
3. **What's the tradeoff?** Balance review volume vs fake detection rate

## Files Added/Modified

### Modified
- [src/image_det/inference/evaluate_test_set.py](../src/image_det/inference/evaluate_test_set.py)
  - Added `samples_per_class_dict` parameter
  - Added `--samples_fake` and `--samples_real` CLI args
  - Enhanced sampling logic with class distribution display

### Added
- [docs/IMBALANCED_EVALUATION.md](../docs/IMBALANCED_EVALUATION.md) - Complete guide
- [examples/evaluate_with_imbalance.py](../examples/evaluate_with_imbalance.py) - Example usage
- [tests/test_imbalanced_sampling.py](../tests/test_imbalanced_sampling.py) - Unit tests

## Example Workflow

```python
# 1. Evaluate with realistic imbalance
metrics, df = evaluate_test_set(
    test_dir="data/image_det/Celeb-DF Preprocessed/test",
    samples_per_class_dict={'fake': 5, 'real': 95}  # 5% fake
)

# 2. Adjust threshold for better detection
from src.image_det.inference.evaluate_test_set import recalculate_metrics_with_threshold

metrics_adj, df_adj = recalculate_metrics_with_threshold(
    df, 
    threshold=0.3,  # Lower = more sensitive
    top_k_percentiles=[5, 10, 20, 30]
)

# 3. Check top-k purity
for k, stats in metrics_adj['top_k_purity'].items():
    print(f"{k}: purity={stats['purity']:.1%}, n={stats['n_samples']}")

# 4. Make decision based on purity and review capacity
# - If top 10% has 80%+ purity → Focus review there
# - If top 20% needed to catch more → Adjust threshold
```

## Backward Compatibility

✅ All existing code continues to work
- `sample_per_class` parameter still works (balanced sampling)
- Default behavior unchanged (no sampling)
- New feature is opt-in via `samples_per_class_dict`

## Testing

Run the unit tests:
```bash
python tests/test_imbalanced_sampling.py
```

Run the examples:
```bash
python examples/evaluate_with_imbalance.py --example both
```

## Documentation

See [IMBALANCED_EVALUATION.md](../docs/IMBALANCED_EVALUATION.md) for:
- Detailed explanation of top-k purity
- Complete API reference
- Real-world use cases
- Decision-making framework
