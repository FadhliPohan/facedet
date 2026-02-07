"""
Report generation and management utilities
"""
import json
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import REPORT_DIR


def generate_report(model_name, model_type, metrics, training_time, 
                   num_samples_train, num_samples_val, class_names,
                   hyperparameters=None, notes=None):
    """
    Generate training report
    
    Args:
        model_name: Name of the model
        model_type: Type of model
        metrics: Dictionary of metrics
        training_time: Training time in seconds
        num_samples_train: Number of training samples
        num_samples_val: Number of validation samples
        class_names: List of class names
        hyperparameters: Dictionary of hyperparameters
        notes: Additional notes
        
    Returns:
        Report dictionary
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    report = {
        'model_info': {
            'model_name': model_name,
            'model_type': model_type,
            'timestamp': timestamp,
            'training_time_seconds': float(training_time),
            'training_time_formatted': format_time(training_time)
        },
        'dataset_info': {
            'num_classes': len(class_names),
            'class_names': class_names,
            'num_samples_train': int(num_samples_train),
            'num_samples_val': int(num_samples_val),
            'total_samples': int(num_samples_train + num_samples_val)
        },
        'metrics': {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score']
        },
        'confusion_matrix': metrics['confusion_matrix'],
        'per_class_metrics': metrics.get('per_class_metrics', {}),
        'hyperparameters': hyperparameters or {},
        'notes': notes or ''
    }
    
    return report


def save_report(report, model_name, confusion_matrix_path=None):
    """
    Save report to JSON file
    
    Args:
        report: Report dictionary
        model_name: Name of the model
        confusion_matrix_path: Path to confusion matrix image
        
    Returns:
        Path to saved report
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"{model_name}_{timestamp}.json"
    report_path = REPORT_DIR / report_filename
    
    # Add confusion matrix path to report
    if confusion_matrix_path:
        report['confusion_matrix_image'] = str(confusion_matrix_path)
    
    # Save report
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"\n{'='*80}")
    print(f"💾 REPORT SAVED")
    print(f"{'='*80}")
    print(f"  Report file: {report_path}")
    if confusion_matrix_path:
        print(f"  Confusion matrix: {confusion_matrix_path}")
    print(f"{'='*80}\n")
    
    return report_path


def load_report(report_path):
    """
    Load report from JSON file
    
    Args:
        report_path: Path to report file
        
    Returns:
        Report dictionary
    """
    with open(report_path, 'r') as f:
        report = json.load(f)
    return report


def list_reports(model_name=None, sort_by='timestamp', reverse=True):
    """
    List all available reports
    
    Args:
        model_name: Filter by model name (optional)
        sort_by: Sort by 'timestamp' or 'accuracy'
        reverse: Sort in reverse order
        
    Returns:
        List of report paths
    """
    report_files = list(REPORT_DIR.glob('*.json'))
    
    # Filter by model name
    if model_name:
        report_files = [f for f in report_files if model_name.lower() in f.stem.lower()]
    
    # Load and sort reports
    reports_with_data = []
    for report_file in report_files:
        try:
            report = load_report(report_file)
            reports_with_data.append((report_file, report))
        except:
            continue
    
    # Sort
    if sort_by == 'timestamp':
        reports_with_data.sort(
            key=lambda x: x[1]['model_info']['timestamp'],
            reverse=reverse
        )
    elif sort_by == 'accuracy':
        reports_with_data.sort(
            key=lambda x: x[1]['metrics']['accuracy'],
            reverse=reverse
        )
    
    return [r[0] for r in reports_with_data]


def display_report(report_path, show_per_class=False):
    """
    Display report in formatted way
    
    Args:
        report_path: Path to report file
        show_per_class: Show per-class metrics
    """
    report = load_report(report_path)
    
    print(f"\n{'='*80}")
    print(f"📋 TRAINING REPORT: {report['model_info']['model_name']}")
    print(f"{'='*80}")
    
    # Model info
    print(f"\n🔹 Model Information:")
    print(f"   Model Type: {report['model_info']['model_type']}")
    print(f"   Timestamp: {report['model_info']['timestamp']}")
    print(f"   Training Time: {report['model_info']['training_time_formatted']}")
    
    # Dataset info
    print(f"\n🔹 Dataset Information:")
    print(f"   Number of Classes: {report['dataset_info']['num_classes']}")
    print(f"   Training Samples: {report['dataset_info']['num_samples_train']:,}")
    print(f"   Validation Samples: {report['dataset_info']['num_samples_val']:,}")
    print(f"   Total Samples: {report['dataset_info']['total_samples']:,}")
    
    # Metrics
    print(f"\n🔹 Performance Metrics:")
    print(f"   Accuracy:  {report['metrics']['accuracy']:6.2f}%")
    print(f"   Precision: {report['metrics']['precision']:6.2f}%")
    print(f"   Recall:    {report['metrics']['recall']:6.2f}%")
    print(f"   F1-Score:  {report['metrics']['f1_score']:6.2f}%")
    
    # Per-class metrics
    if show_per_class and 'per_class_metrics' in report:
        print(f"\n🔹 Per-Class Metrics:")
        print(f"   {'Class':<50} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
        print("   " + "-" * 80)
        for class_name, metrics in report['per_class_metrics'].items():
            print(f"   {class_name[:48]:<50} "
                  f"{metrics['precision']:>9.2f}% "
                  f"{metrics['recall']:>9.2f}% "
                  f"{metrics['f1_score']:>9.2f}%")
    
    # Hyperparameters
    if report.get('hyperparameters'):
        print(f"\n🔹 Hyperparameters:")
        for key, value in report['hyperparameters'].items():
            print(f"   {key}: {value}")
    
    # Notes
    if report.get('notes'):
        print(f"\n🔹 Notes:")
        print(f"   {report['notes']}")
    
    # Confusion matrix
    if 'confusion_matrix_image' in report:
        print(f"\n🔹 Confusion Matrix:")
        print(f"   Image: {report['confusion_matrix_image']}")
    
    print(f"\n{'='*80}\n")


def format_time(seconds):
    """
    Format seconds to human-readable time
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def get_latest_report(model_name=None):
    """
    Get the latest report for a model
    
    Args:
        model_name: Model name to filter (optional)
        
    Returns:
        Path to latest report or None
    """
    reports = list_reports(model_name=model_name, sort_by='timestamp', reverse=True)
    return reports[0] if reports else None
