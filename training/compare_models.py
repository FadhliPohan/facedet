#!/usr/bin/env python3
"""
Model Comparison Tool
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

sys.path.append(str(Path(__file__).parent))

from config import REPORT_DIR
from utils.report_generator import list_reports, load_report


def compare_models(show_per_class=False, save_comparison=True):
    """
    Compare all trained models
    
    Args:
        show_per_class: Show per-class comparison
        save_comparison: Save comparison to file
    """
    print("\n" + "="*80)
    print("📊 MODEL COMPARISON")
    print("="*80 + "\n")
    
    # Get all reports
    report_paths = list_reports(sort_by='timestamp', reverse=False)
    
    if not report_paths:
        print("❌ No training reports found!")
        print(f"   Please train at least one model first.")
        return
    
    print(f"Found {len(report_paths)} training report(s)\n")
    
    # Load all reports
    models_data = []
    for report_path in report_paths:
        try:
            report = load_report(report_path)
            models_data.append({
                'Model': report['model_info']['model_name'],
                'Type': report['model_info']['model_type'],
                'Accuracy': report['metrics']['accuracy'],
                'Precision': report['metrics']['precision'],
                'Recall': report['metrics']['recall'],
                'F1-Score': report['metrics']['f1_score'],
                'Training Time': report['model_info']['training_time_seconds'],
                'Time Formatted': report['model_info']['training_time_formatted'],
                'Timestamp': report['model_info']['timestamp'],
                'Report': report
            })
        except Exception as e:
            print(f"Warning: Could not load {report_path}: {e}")
    
    if not models_data:
        print("❌ No valid training reports found!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(models_data)
    
    # Sort by accuracy
    df = df.sort_values('Accuracy', ascending=False)
    
    # Display comparison table
    print("="*120)
    print("OVERALL MODEL COMPARISON")
    print("="*120)
    print(f"{'Rank':<6} {'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Training Time':>15}")
    print("-"*120)
    
    for idx, row in enumerate(df.itertuples(), 1):
        print(f"{idx:<6} {row.Model:<25} {row.Accuracy:>9.2f}% {row.Precision:>9.2f}% "
              f"{row.Recall:>9.2f}% {getattr(row, 'F1-Score'):>9.2f}% {row._9:>15}")
    
    print("="*120)
    
    # Best model
    best_model = df.iloc[0]
    print(f"\n🏆 Best Model: {best_model['Model']}")
    print(f"   Accuracy: {best_model['Accuracy']:.2f}%")
    print(f"   F1-Score: {best_model['F1-Score']:.2f}%")
    print(f"   Training Time: {best_model['Time Formatted']}")
    
    # Visualizations
    print("\n📊 Generating comparison visualizations...")
    
    # 1. Metrics comparison bar chart
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for idx, (ax, metric, color) in enumerate(zip(axes.flat, metrics, colors)):
        df_sorted = df.sort_values(metric, ascending=True)
        ax.barh(df_sorted['Model'], df_sorted[metric], color=color, alpha=0.7)
        ax.set_xlabel(f'{metric} (%)', fontsize=11)
        ax.set_title(f'{metric} Comparison', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add values on bars
        for i, v in enumerate(df_sorted[metric]):
            ax.text(v + 0.5, i, f'{v:.2f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    comparison_chart_path = REPORT_DIR / "model_comparison_metrics.png"
    plt.savefig(comparison_chart_path, dpi=300, bbox_inches='tight')
    print(f"✅ Metrics comparison saved to: {comparison_chart_path}")
    plt.close()
    
    # 2. Overall performance radar chart
    if len(models_data) <= 6:  # Only if not too many models
        create_radar_chart(df, REPORT_DIR / "model_comparison_radar.png")
    
    # 3. Training time comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    df_sorted = df.sort_values('Training Time', ascending=True)
    bars = ax.barh(df_sorted['Model'], df_sorted['Training Time'] / 60, color='#9b59b6', alpha=0.7)
    ax.set_xlabel('Training Time (minutes)', fontsize=12)
    ax.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add values on bars
    for i, (v, time_str) in enumerate(zip(df_sorted['Training Time'], df_sorted['Time Formatted'])):
        ax.text(v/60 + 0.5, i, time_str, va='center', fontsize=9)
    
    plt.tight_layout()
    time_chart_path = REPORT_DIR / "model_comparison_time.png"
    plt.savefig(time_chart_path, dpi=300, bbox_inches='tight')
    print(f"✅ Training time comparison saved to: {time_chart_path}")
    plt.close()
    
    # Save comparison to JSON
    if save_comparison:
        comparison_data = {
            'comparison_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'num_models': len(models_data),
            'models': []
        }
        
        for _, row in df.iterrows():
            comparison_data['models'].append({
                'model_name': row['Model'],
                'model_type': row['Type'],
                'accuracy': row['Accuracy'],
                'precision': row['Precision'],
                'recall': row['Recall'],
                'f1_score': row['F1-Score'],
                'training_time': row['Time Formatted'],
                'timestamp': row['Timestamp']
            })
        
        comparison_path = REPORT_DIR / f"model_comparison_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(comparison_path, 'w') as f:
            json.dump(comparison_data, f, indent=4)
        print(f"✅ Comparison data saved to: {comparison_path}")
    
    print("\n" + "="*80 + "\n")
    
    return df


def create_radar_chart(df, save_path):
    """Create radar chart for model comparison"""
    import numpy as np
    
    # Prepare data
    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    num_vars = len(categories)
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    # Initialize plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot each model
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    for idx, row in enumerate(df.itertuples()):
        values = [row.Accuracy, row.Precision, row.Recall, getattr(row, 'F1-Score')]
        values += values[:1]
        
        color = colors[idx % len(colors)]
        ax.plot(angles, values, 'o-', linewidth=2, label=row.Model, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    # Fix axis to go in the right order
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    
    # Set y-axis limits
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=9)
    ax.grid(True)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.title('Model Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Radar chart saved to: {save_path}")
    plt.close()


if __name__ == "__main__":
    compare_models(save_comparison=True)
