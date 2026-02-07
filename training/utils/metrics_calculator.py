"""
Metrics calculation and visualization utilities
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from pathlib import Path


def calculate_metrics(y_true, y_pred, class_names=None, verbose=True):
    """
    Calculate classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        verbose: Print metrics
        
    Returns:
        Dictionary containing all metrics
    """
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0) * 100
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0) * 100
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0) * 100
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'per_class_metrics': {}
    }
    
    # Add per-class metrics
    if class_names:
        for i, class_name in enumerate(class_names):
            metrics['per_class_metrics'][class_name] = {
                'precision': float(precision_per_class[i]) if i < len(precision_per_class) else 0.0,
                'recall': float(recall_per_class[i]) if i < len(recall_per_class) else 0.0,
                'f1_score': float(f1_per_class[i]) if i < len(f1_per_class) else 0.0
            }
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"📊 EVALUATION METRICS")
        print(f"{'='*80}")
        print(f"  Accuracy:  {accuracy:6.2f}%")
        print(f"  Precision: {precision:6.2f}%")
        print(f"  Recall:    {recall:6.2f}%")
        print(f"  F1-Score:  {f1:6.2f}%")
        print(f"{'='*80}\n")
        
        if class_names:
            print("Per-class metrics:")
            print(f"{'Class':<50} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
            print("-" * 80)
            for class_name in class_names:
                metrics_data = metrics['per_class_metrics'][class_name]
                print(f"{class_name[:48]:<50} "
                      f"{metrics_data['precision']:>9.2f}% "
                      f"{metrics_data['recall']:>9.2f}% "
                      f"{metrics_data['f1_score']:>9.2f}%")
            print()
    
    return metrics


def plot_confusion_matrix(cm, class_names, save_path=None, figsize=(12, 10), cmap='Blues'):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save the plot
        figsize: Figure size
        cmap: Colormap
        
    Returns:
        Path to saved plot
    """
    # Convert to numpy array if it's a list
    if isinstance(cm, list):
        cm = np.array(cm)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot confusion matrix
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap=cmap,
        xticklabels=[name[:20] for name in class_names],
        yticklabels=[name[:20] for name in class_names],
        cbar_kws={'label': 'Count'}
    )
    
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved to: {save_path}")
    
    plt.close()
    
    return save_path


def plot_training_history(history, save_path=None):
    """
    Plot training history (for CNN model)
    
    Args:
        history: Keras training history object
        save_path: Path to save the plot
        
    Returns:
        Path to saved plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training history saved to: {save_path}")
    
    plt.close()
    
    return save_path
