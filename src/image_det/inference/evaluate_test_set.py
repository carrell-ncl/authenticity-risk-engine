#!/usr/bin/env python
"""
Inference script to evaluate the best trained deepfake detection model on test set.

This script:
  1. Loads the best checkpoint from the most recent training run
  2. Processes test images from data/image_det/Celeb-DF Preprocessed/test
  3. Computes predictions and accuracy metrics
  4. Supports sampling to create imbalanced test sets for real-world simulation

Usage:
  # Evaluate on full test set
  python evaluate_test_set.py --test_dir "data/image_det/Celeb-DF Preprocessed/test"
  
  # Use specific checkpoint
  python evaluate_test_set.py --checkpoint_path "models/image_det/runs_deepfake/20260116_171601/best.pt"
  
  # Balanced sampling - 50 from each class
  python evaluate_test_set.py --sample_per_class 50
  
  # Imbalanced sampling - simulate real-world where only 10% are fake
  python evaluate_test_set.py --samples_fake 10 --samples_real 90
  
  # Or another realistic scenario - 5% fake
  python evaluate_test_set.py --samples_fake 5 --samples_real 95

In Python/Notebook:
  from src.image_det.inference.evaluate_test_set import evaluate_test_set
  
  # Balanced
  metrics, df = evaluate_test_set(test_dir, sample_per_class=50)
  
  # Imbalanced - 10% fake
  metrics, df = evaluate_test_set(test_dir, samples_per_class_dict={'fake': 10, 'real': 90})
"""

import os
import json
import glob
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    classification_report
)
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


def build_model(model_name: str) -> nn.Module:
    """Build model architecture matching training script."""
    if model_name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)  # No pretrained weights, will load from checkpoint
        in_features = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_features, 1)
        return m

    if model_name == "resnet50":
        m = models.resnet50(weights=None)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, 1)
        return m

    raise ValueError(f"Unsupported model_name={model_name}. Use efficientnet_b0 or resnet50.")


def find_latest_checkpoint(runs_dir: str) -> str:
    """Find the most recent best.pt checkpoint in runs directory."""
    checkpoint_pattern = os.path.join(runs_dir, "*/best.pt")
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {runs_dir}")
    
    # Sort by modification time, most recent first
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    return checkpoints[0]


def load_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load model and config from checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    cfg = checkpoint.get("cfg", {})
    model_name = cfg.get("model_name", "efficientnet_b0")
    
    model = build_model(model_name)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()
    
    print(f"Model architecture: {model_name}")
    print(f"Trained for {checkpoint.get('epoch', 'unknown')} epochs")
    
    return model, cfg


def get_test_transforms(img_size: int = 224) -> transforms.Compose:
    """Get test-time image transforms."""
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference on all images in loader.
    
    Returns:
        predictions: Binary predictions (0 or 1)
        probabilities: Probability scores for class 1 (real)
        targets: Ground truth labels
    """
    model.eval()
    
    all_probs = []
    all_targets = []
    
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).long()
        
        logits = model(images)
        probs = torch.sigmoid(logits.squeeze(1))
        
        all_probs.append(probs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
    
    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    all_preds = (all_probs >= 0.5).astype(int)
    
    return all_preds, all_probs, all_targets


def create_results_dataframe(
    predictions: np.ndarray,
    probabilities: np.ndarray,
    targets: np.ndarray,
    image_paths: List[str],
    class_names: List[str]
) -> pd.DataFrame:
    """
    Create a pandas DataFrame with detailed prediction results.
    
    Args:
        predictions: Binary predictions (0 or 1)
        probabilities: Probability scores for class 1
        targets: Ground truth labels
        image_paths: List of image file paths
        class_names: List of class names ['fake', 'real'] or similar
    
    Returns:
        DataFrame with columns: image_path, filename, true_label, predicted_label,
                               probability_fake, probability_real, is_correct
    """
    df = pd.DataFrame({
        'image_path': image_paths,
        'filename': [os.path.basename(p) for p in image_paths],
        'true_label': [class_names[t] for t in targets],
        'true_label_idx': targets,
        'predicted_label': [class_names[p] for p in predictions],
        'predicted_label_idx': predictions,
        'probability_fake': 1 - probabilities,
        'probability_real': probabilities,
        'is_correct': predictions == targets
    })
    
    # Add confidence (max probability)
    df['confidence'] = df[['probability_fake', 'probability_real']].max(axis=1)
    
    return df


def compute_metrics(predictions: np.ndarray, probabilities: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
    """Compute comprehensive evaluation metrics."""
    
    # Basic metrics
    accuracy = accuracy_score(targets, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average='binary')
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(targets, predictions).ravel()
    
    # ROC AUC
    try:
        roc_auc = roc_auc_score(targets, probabilities)
    except ValueError:
        roc_auc = None
    
    # Per-class metrics
    class_report = classification_report(targets, predictions, target_names=['fake', 'real'], output_dict=True)
    
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "class_metrics": {
            "fake": {
                "precision": class_report['fake']['precision'],
                "recall": class_report['fake']['recall'],
                "f1-score": class_report['fake']['f1-score'],
                "support": int(class_report['fake']['support'])
            },
            "real": {
                "precision": class_report['real']['precision'],
                "recall": class_report['real']['recall'],
                "f1-score": class_report['real']['f1-score'],
                "support": int(class_report['real']['support'])
            }
        },
        "total_samples": len(targets)
    }
    
    return metrics


def recalculate_metrics_with_threshold(
    df_results: pd.DataFrame,
    threshold: float = 0.5,
    class_names: Optional[List[str]] = None,
    top_k_percentiles: Optional[List[int]] = None
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Recalculate metrics using a different classification threshold.
    
    Args:
        df_results: DataFrame from evaluate_test_set with predictions
        threshold: New threshold for classification (default: 0.5)
                  Values >= threshold are classified as 'real', < threshold as 'fake'
        class_names: List of class names (default: ['fake', 'real'])
        top_k_percentiles: List of percentiles for top-k purity analysis (default: [10, 20, 30])
                          e.g. [10, 20, 30] calculates accuracy for top 10%, 20%, 30% most confident predictions
    
    Returns:
        metrics: Updated metrics dictionary with new threshold and top-k purity
        df_updated: Updated DataFrame with new predictions
    
    Example:
        # Get results with default threshold (0.5)
        metrics, df = evaluate_test_set(test_dir)
        
        # Try lower threshold to catch more fakes (higher sensitivity)
        metrics_03, df_03 = recalculate_metrics_with_threshold(df, threshold=0.3)
        
        # Try higher threshold to reduce false positives (higher specificity)
        metrics_07, df_07 = recalculate_metrics_with_threshold(df, threshold=0.7)
    """
    
    if class_names is None:
        class_names = ['fake', 'real']
    
    if top_k_percentiles is None:
        top_k_percentiles = [10, 20, 30]
    
    # Create copy to avoid modifying original
    df_updated = df_results.copy()
    
    # Recalculate predictions with new threshold
    new_predictions = (df_updated['probability_real'] >= threshold).astype(int)
    df_updated['predicted_label'] = [class_names[p] for p in new_predictions]
    df_updated['predicted_label_idx'] = new_predictions
    df_updated['is_correct'] = new_predictions == df_updated['true_label_idx']
    df_updated['confidence'] = df_updated[['probability_fake', 'probability_real']].max(axis=1)
    
    # Compute new metrics
    metrics = compute_metrics(
        predictions=new_predictions,
        probabilities=df_updated['probability_real'].values,
        targets=df_updated['true_label_idx'].values
    )
    
    # Add threshold info
    metrics['threshold'] = threshold
    metrics['class_names'] = class_names
    
    # Calculate top-k purity metrics
    # Sort by confidence descending to get most confident predictions first
    df_sorted = df_updated.sort_values('confidence', ascending=False).reset_index(drop=True)
    
    top_k_purity = {}
    for k in top_k_percentiles:
        # Calculate how many samples to take (top k%)
        n_samples = max(1, int(len(df_sorted) * k / 100))
        top_k_samples = df_sorted.head(n_samples)
        
        # Calculate accuracy (purity) of top k% predictions
        purity = top_k_samples['is_correct'].mean()
        
        # Also calculate class distribution in top k
        pred_dist = top_k_samples['predicted_label'].value_counts(normalize=True).to_dict()
        true_dist = top_k_samples['true_label'].value_counts(normalize=True).to_dict()
        
        top_k_purity[f'top_{k}'] = {
            'purity': float(purity),
            'accuracy': float(purity),  # Same as purity
            'n_samples': int(n_samples),
            'min_confidence': float(top_k_samples['confidence'].min()),
            'mean_confidence': float(top_k_samples['confidence'].mean()),
            'predicted_distribution': {k: float(v) for k, v in pred_dist.items()},
            'true_distribution': {k: float(v) for k, v in true_dist.items()}
        }
    
    metrics['top_k_purity'] = top_k_purity
    
    return metrics, df_updated


def find_optimal_threshold(
    df_results: pd.DataFrame,
    metric: str = 'f1_score',
    thresholds: Optional[np.ndarray] = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Find optimal classification threshold by maximizing a specific metric.
    
    Args:
        df_results: DataFrame from evaluate_test_set
        metric: Metric to optimize ('f1_score', 'accuracy', 'recall', 'precision')
        thresholds: Array of thresholds to test (default: np.arange(0.1, 1.0, 0.05))
    
    Returns:
        best_threshold: Optimal threshold value
        best_metrics: Metrics at optimal threshold
    
    Example:
        metrics, df = evaluate_test_set(test_dir)
        best_threshold, best_metrics = find_optimal_threshold(df, metric='f1_score')
        print(f"Optimal threshold: {best_threshold:.2f}")
        print(f"F1-Score: {best_metrics['f1_score']:.4f}")
    """
    
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)
    
    best_score = -1
    best_threshold = 0.5
    best_metrics = None
    
    for thresh in thresholds:
        metrics, _ = recalculate_metrics_with_threshold(df_results, threshold=thresh)
        score = metrics.get(metric, 0)
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
            best_metrics = metrics
    
    return best_threshold, best_metrics


def plot_threshold_analysis(
    df_results: pd.DataFrame,
    thresholds: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> None:
    """
    Plot how metrics change across different thresholds.
    
    Args:
        df_results: DataFrame from evaluate_test_set
        thresholds: Array of thresholds to test
        figsize: Figure size
    
    Example:
        metrics, df = evaluate_test_set(test_dir)
        plot_threshold_analysis(df)
    """
    
    if not PLOTTING_AVAILABLE:
        print("Warning: matplotlib/seaborn not available. Cannot plot.")
        return
    
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.02)
    
    # Calculate metrics for each threshold
    results = {
        'threshold': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': []
    }
    
    for thresh in thresholds:
        metrics, _ = recalculate_metrics_with_threshold(df_results, threshold=thresh)
        results['threshold'].append(thresh)
        results['accuracy'].append(metrics['accuracy'])
        results['precision'].append(metrics['precision'])
        results['recall'].append(metrics['recall'])
        results['f1_score'].append(metrics['f1_score'])
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: All metrics
    ax = axes[0, 0]
    ax.plot(results['threshold'], results['accuracy'], 'o-', label='Accuracy', linewidth=2)
    ax.plot(results['threshold'], results['precision'], 's-', label='Precision', linewidth=2)
    ax.plot(results['threshold'], results['recall'], '^-', label='Recall', linewidth=2)
    ax.plot(results['threshold'], results['f1_score'], 'd-', label='F1-Score', linewidth=2)
    ax.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Default (0.5)')
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Metrics vs Threshold', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 2: Precision-Recall tradeoff
    ax = axes[0, 1]
    ax.plot(results['recall'], results['precision'], 'o-', linewidth=2)
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Plot 3: F1-Score
    ax = axes[1, 0]
    ax.plot(results['threshold'], results['f1_score'], 'o-', linewidth=2, color='green')
    best_idx = np.argmax(results['f1_score'])
    ax.axvline(results['threshold'][best_idx], color='orange', linestyle='--', 
               label=f"Best: {results['threshold'][best_idx]:.2f}")
    ax.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Default (0.5)')
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('F1-Score vs Threshold', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 4: Confusion matrix metrics
    ax = axes[1, 1]
    # Calculate TPR and FPR for each threshold
    tpr_list = []
    fpr_list = []
    for thresh in thresholds:
        metrics, _ = recalculate_metrics_with_threshold(df_results, threshold=thresh)
        cm = metrics['confusion_matrix']
        tpr = cm['true_positives'] / (cm['true_positives'] + cm['false_negatives'])
        fpr = cm['false_positives'] / (cm['false_positives'] + cm['true_negatives'])
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    ax.plot(results['threshold'], tpr_list, 'o-', label='TPR (Sensitivity)', linewidth=2)
    ax.plot(results['threshold'], fpr_list, 's-', label='FPR (False Alarm)', linewidth=2)
    ax.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Default (0.5)')
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Rate', fontsize=12)
    ax.set_title('TPR and FPR vs Threshold', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print optimal thresholds
    print("\n" + "="*60)
    print("OPTIMAL THRESHOLDS")
    print("="*60)
    for metric_name in ['accuracy', 'precision', 'recall', 'f1_score']:
        best_idx = np.argmax(results[metric_name])
        print(f"\nBest {metric_name}: {results[metric_name][best_idx]:.4f} at threshold {results['threshold'][best_idx]:.2f}")
    print("="*60)


def print_metrics(metrics: Dict[str, Any]) -> None:
    """Pretty print evaluation metrics."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    if metrics['roc_auc'] is not None:
        print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
    
    print(f"\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"                    Predicted")
    print(f"                Fake      Real")
    print(f"  Actual Fake  {cm['true_negatives']:5d}     {cm['false_positives']:5d}")
    print(f"         Real  {cm['false_negatives']:5d}     {cm['true_positives']:5d}")
    print(f"\n  True Negatives (Fake correctly classified):  {cm['true_negatives']}")
    print(f"  False Positives (Fake predicted as Real):    {cm['false_positives']}")
    print(f"  False Negatives (Real predicted as Fake):    {cm['false_negatives']}")
    print(f"  True Positives (Real correctly classified):  {cm['true_positives']}")
    
    print(f"\nPer-Class Metrics:")
    for class_name, class_metrics in metrics['class_metrics'].items():
        print(f"  {class_name.upper()}:")
        print(f"    Precision: {class_metrics['precision']:.4f}")
        print(f"    Recall:    {class_metrics['recall']:.4f}")
        print(f"    F1-Score:  {class_metrics['f1-score']:.4f}")
        print(f"    Support:   {class_metrics['support']}")
    
    print(f"\nTotal Samples: {metrics['total_samples']}")
    print("="*60 + "\n")


def plot_confusion_matrix(
    metrics: Dict[str, Any],
    class_names: List[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = 'Blues',
    save_path: Optional[str] = None
) -> None:
    """
    Plot confusion matrix using matplotlib/seaborn.
    
    Args:
        metrics: Metrics dictionary containing confusion_matrix
        class_names: List of class names (default: ['Fake', 'Real'])
        figsize: Figure size
        cmap: Colormap for heatmap
        save_path: If provided, save the plot to this path
    """
    if not PLOTTING_AVAILABLE:
        print("Warning: matplotlib/seaborn not available. Cannot plot confusion matrix.")
        return
    
    if class_names is None:
        class_names = ['Fake', 'Real']
    
    cm = metrics['confusion_matrix']
    confusion_array = np.array([
        [cm['true_negatives'], cm['false_positives']],
        [cm['false_negatives'], cm['true_positives']]
    ])
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        confusion_array,
        annot=True,
        fmt='d',
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'},
        square=True
    )
    
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Add accuracy text
    accuracy = metrics['accuracy']
    plt.text(
        0.5, -0.15,
        f"Overall Accuracy: {accuracy:.2%}",
        ha='center',
        va='center',
        transform=plt.gca().transAxes,
        fontsize=11,
        fontweight='bold'
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()


def evaluate_test_set(
    test_dir: str,
    checkpoint_path: Optional[str] = None,
    runs_dir: str = r"models\image_det\runs_deepfake",
    batch_size: int = 32,
    num_workers: int = 4,
    return_dataframe: bool = True,
    verbose: bool = True,
    sample_per_class: Optional[int] = None,
    samples_per_class_dict: Optional[Dict[str, int]] = None,
    seed: int = 42
) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
    """
    Evaluate trained model on test set. Notebook-friendly function.
    
    Args:
        test_dir: Path to test directory with fake/ and real/ subfolders
        checkpoint_path: Path to specific checkpoint (None = use latest)
        runs_dir: Directory containing training runs
        batch_size: Batch size for inference
        num_workers: Number of data loading workers
        return_dataframe: If True, return DataFrame with detailed results
        verbose: If True, print progress and results
        sample_per_class: If specified, randomly sample this many images from each class (deprecated - use samples_per_class_dict)
        samples_per_class_dict: Dict mapping class names to sample counts, e.g., {'fake': 10, 'real': 90} for realistic imbalance
        seed: Random seed for sampling
    
    Returns:
        metrics: Dictionary with evaluation metrics
        df_results: DataFrame with per-image predictions (if return_dataframe=True)
    
    Examples:
        # Balanced sampling - 50 from each class
        evaluate_test_set(test_dir, sample_per_class=50)
        
        # Imbalanced sampling - simulate real-world where only 10% are fake
        evaluate_test_set(test_dir, samples_per_class_dict={'fake': 10, 'real': 90})
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")
    
    # Find checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint(runs_dir)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load model
    model, cfg = load_checkpoint(checkpoint_path, device)
    img_size = cfg.get("img_size", 224)
    
    # Prepare test dataset
    if verbose:
        print(f"\nLoading test data from: {test_dir}")
    
    test_transforms = get_test_transforms(img_size)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    if verbose:
        print(f"Classes: {test_dataset.classes}")
        print(f"Class to index mapping: {test_dataset.class_to_idx}")
        print(f"Total test samples: {len(test_dataset)}")
    
    # Sample subset if requested
    if samples_per_class_dict is not None or sample_per_class is not None:
        # Group indices by class
        class_indices = {}
        for idx, (_, label) in enumerate(test_dataset.samples):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        
        # Determine samples per class
        np.random.seed(seed)
        sampled_indices = []
        
        if samples_per_class_dict is not None:
            # Use custom sampling per class
            if verbose:
                print(f"\nSampling with custom distribution:")
            
            for label, indices in class_indices.items():
                class_name = test_dataset.classes[label]
                n_to_sample = samples_per_class_dict.get(class_name, 0)
                n_samples = min(n_to_sample, len(indices))
                
                if n_samples > 0:
                    sampled = np.random.choice(indices, size=n_samples, replace=False)
                    sampled_indices.extend(sampled)
                
                if verbose:
                    print(f"  {class_name}: sampled {n_samples}/{len(indices)} (requested: {n_to_sample})")
        else:
            # Use uniform sampling (backward compatibility)
            if verbose:
                print(f"\nSampling {sample_per_class} images per class...")
            
            for label, indices in class_indices.items():
                n_samples = min(sample_per_class, len(indices))
                sampled = np.random.choice(indices, size=n_samples, replace=False)
                sampled_indices.extend(sampled)
                if verbose:
                    print(f"  {test_dataset.classes[label]}: sampled {n_samples}/{len(indices)}")
        
        # Create subset
        from torch.utils.data import Subset
        test_dataset = Subset(test_dataset, sampled_indices)
        
        if verbose:
            print(f"Total sampled: {len(test_dataset)}")
            if samples_per_class_dict is not None:
                # Calculate and display class distribution
                class_counts = {}
                for idx in sampled_indices:
                    _, label = test_dataset.dataset.samples[idx]
                    class_name = test_dataset.dataset.classes[label]
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
                print(f"Class distribution:")
                for class_name, count in class_counts.items():
                    pct = 100 * count / len(sampled_indices)
                    print(f"  {class_name}: {count} ({pct:.1f}%)")
    
    # Get image paths (handle both full dataset and Subset)
    if hasattr(test_dataset, 'samples'):
        image_paths = [s[0] for s in test_dataset.samples]
    else:
        # It's a Subset
        base_samples = test_dataset.dataset.samples
        image_paths = [base_samples[idx][0] for idx in test_dataset.indices]
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )
    
    # Run predictions
    if verbose:
        print("\nRunning inference...")
    predictions, probabilities, targets = predict(model, test_loader, device)
    
    # Compute metrics
    if verbose:
        print("Computing metrics...")
    metrics = compute_metrics(predictions, probabilities, targets)
    
    # Add metadata
    metrics["checkpoint_path"] = checkpoint_path
    metrics["test_dir"] = test_dir
    metrics["model_name"] = cfg.get("model_name", "unknown")
    if hasattr(test_dataset, 'classes'):
        metrics["classes"] = test_dataset.classes
        metrics["class_to_idx"] = test_dataset.class_to_idx
    else:
        # It's a Subset
        metrics["classes"] = test_dataset.dataset.classes
        metrics["class_to_idx"] = test_dataset.dataset.class_to_idx
    metrics["sample_per_class"] = sample_per_class
    metrics["samples_per_class_dict"] = samples_per_class_dict
    metrics["seed"] = seed if (sample_per_class or samples_per_class_dict) else None
    
    # Print results
    if verbose:
        print_metrics(metrics)
    
    # Create DataFrame
    df_results = None
    if return_dataframe:
        # Get class names
        if hasattr(test_dataset, 'classes'):
            class_names = test_dataset.classes
        else:
            class_names = test_dataset.dataset.classes
        
        df_results = create_results_dataframe(
            predictions=predictions,
            probabilities=probabilities,
            targets=targets,
            image_paths=image_paths,
            class_names=class_names
        )
        if verbose:
            print(f"\nDataFrame created with {len(df_results)} rows")
    
    return metrics, df_results


def main(
    test_dir: str,
    checkpoint_path: str = None,
    runs_dir: str = "models/image_det/runs_deepfake",
    batch_size: int = 32,
    num_workers: int = 4,
    output_file: str = None,
    sample_per_class: int = None,
    samples_per_class_dict: Dict[str, int] = None,
    seed: int = 42
) -> None:
    """Main evaluation function for CLI usage."""
    
    # Run evaluation
    metrics, df_results = evaluate_test_set(
        test_dir=test_dir,
        checkpoint_path=checkpoint_path,
        runs_dir=runs_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        return_dataframe=True,
        verbose=True,
        sample_per_class=sample_per_class,
        samples_per_class_dict=samples_per_class_dict,
        seed=seed
    )
    
    # Save results
    if output_file is None:
        checkpoint_dir = os.path.dirname(metrics["checkpoint_path"])
        output_file = os.path.join(checkpoint_dir, "test_results.json")
    
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    
    # Save metrics as JSON
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nMetrics saved to: {output_file}")
    
    # Save DataFrame as CSV
    if df_results is not None:
        csv_file = output_file.replace('.json', '_predictions.csv')
        df_results.to_csv(csv_file, index=False)
        print(f"Predictions saved to: {csv_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained deepfake detection model on test set")
    parser.add_argument(
        "--test_dir",
        type=str,
        default=r"data\image_det\Celeb-DF Preprocessed\test",
        help="Path to test directory containing fake/ and real/ subfolders"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to specific checkpoint (default: use most recent from runs_dir)"
    )
    parser.add_argument(
        "--runs_dir",
        type=str,
        default=r"models\image_det\runs_deepfake",
        help="Directory containing training runs (used if checkpoint_path not specified)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save results JSON (default: save in checkpoint directory)"
    )
    parser.add_argument(
        "--sample_per_class",
        type=int,
        default=None,
        help="Randomly sample this many images from each class (default: use all images)"
    )
    parser.add_argument(
        "--samples_fake",
        type=int,
        default=None,
        help="Number of fake samples to use (for imbalanced evaluation). Use with --samples_real"
    )
    parser.add_argument(
        "--samples_real",
        type=int,
        default=None,
        help="Number of real samples to use (for imbalanced evaluation). Use with --samples_fake"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling"
    )
    
    args = parser.parse_args()
    
    # Build samples_per_class_dict if individual class samples specified
    samples_per_class_dict = None
    if args.samples_fake is not None or args.samples_real is not None:
        samples_per_class_dict = {}
        if args.samples_fake is not None:
            samples_per_class_dict['fake'] = args.samples_fake
        if args.samples_real is not None:
            samples_per_class_dict['real'] = args.samples_real
    
    main(
        test_dir=args.test_dir,
        checkpoint_path=args.checkpoint_path,
        runs_dir=args.runs_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_file=args.output_file,
        sample_per_class=args.sample_per_class,
        samples_per_class_dict=samples_per_class_dict,
        seed=args.seed
    )
