#!/usr/bin/env python3
"""
View Training Reports
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from utils.report_generator import list_reports, display_report


def view_reports(model_name=None, show_per_class=True):
    """
    View training reports
    
    Args:
        model_name: Filter by model name (optional)
        show_per_class: Show per-class metrics
    """
    print("\n" + "="*80)
    print("📋 TRAINING REPORTS")
    print("="*80 + "\n")
    
    # Get all reports
    report_paths = list_reports(model_name=model_name, sort_by='timestamp', reverse=True)
    
    if not report_paths:
        print("❌ No training reports found!")
        if model_name:
            print(f"   No reports found for model: {model_name}")
        else:
            print(f"   Please train at least one model first.")
        return
    
    print(f"Found {len(report_paths)} training report(s)\n")
    
    # Display all reports
    for idx, report_path in enumerate(report_paths, 1):
        print(f"\n{'#'*80}")
        print(f"# REPORT {idx}/{len(report_paths)}")
        print(f"{'#'*80}")
        display_report(report_path, show_per_class=show_per_class)
        
        if idx < len(report_paths):
            input("Press Enter to view next report...")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='View training reports')
    parser.add_argument('--model', type=str, help='Filter by model name', default=None)
    parser.add_argument('--no-per-class', action='store_true', help='Hide per-class metrics')
    
    args = parser.parse_args()
    
    view_reports(model_name=args.model, show_per_class=not args.no_per_class)
