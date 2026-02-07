#!/usr/bin/env python3
"""
Main Training Orchestrator
Interactive menu system for training, testing, and comparing models
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
from pathlib import Path
import tensorflow as tf

sys.path.append(str(Path(__file__).parent))

from config import REPORT_DIR, MODEL_DIR
from utils.report_generator import list_reports, get_latest_report, display_report


def print_banner():
    """Print application banner"""
    print("\n" + "="*80)
    print("🤖 SKIN DISEASE DETECTION - TRAINING SYSTEM")
    print("="*80)
    print("   Multiple Model Training & Comparison Tool")
    print("   Models: ResNet50, VGG16, MobileNetV2, InceptionV3, DenseNet121, CNN")
    print("="*80 + "\n")


def check_existing_reports():
    """Check for existing training reports"""
    reports = list_reports()
    
    if reports:
        print(f"📊 Found {len(reports)} existing training report(s)")
        print("\nLatest reports:")
        
        # Show latest 5 reports
        for report_path in reports[:5]:
            from utils.report_generator import load_report
            report = load_report(report_path)
            print(f"  - {report['model_info']['model_name']}: "
                  f"{report['metrics']['accuracy']:.2f}% accuracy "
                  f"({report['model_info']['timestamp']})")
        
        if len(reports) > 5:
            print(f"  ... and {len(reports) - 5} more")
        
        print()
        return True
    else:
        print("📊 No existing training reports found")
        print("   You can start training models from scratch\n")
        return False


def train_single_model(model_choice):
    """Train a single model"""
    models = {
        '1': ('ResNet50 + SVM', 'resnet50_svm'),
        '2': ('VGG16 + SVM', 'vgg16_svm'),
        '3': ('MobileNetV2 + SVM', 'MobileNetV2_svm'),
        '4': ('InceptionV3 + SVM', 'InceptionV3_svm'),
        '5': ('DenseNet121 + SVM', 'DenseNet121_svm'),
        '6': ('CNN + Softmax', 'cnn_softmax')
    }
    
    if model_choice not in models:
        print("❌ Invalid model choice!")
        return
    
    model_name, script_name = models[model_choice]
    
    print(f"\n🚀 Starting training: {model_name}")
    print("="*80 + "\n")
    
    # Check if already trained
    latest_report = get_latest_report(model_name.replace(' + ', '_').replace(' ', '_'))
    
    if latest_report:
        from utils.report_generator import load_report
        report = load_report(latest_report)
        
        print(f"⚠️  Found existing training for {model_name}:")
        print(f"   Accuracy: {report['metrics']['accuracy']:.2f}%")
        print(f"   Trained on: {report['model_info']['timestamp']}\n")
        
        response = input("Do you want to retrain from scratch? (y/n): ").lower()
        if response != 'y':
            print("Skipping training...")
            return
        print("\nRetraining model...\n")
    
    # Import and run training script
    try:
        if script_name == 'resnet50_svm':
            from resnet50_svm import train_resnet50_svm
            train_resnet50_svm()
        elif script_name == 'vgg16_svm':
            from vgg16_svm import train_vgg16_svm
            train_vgg16_svm()
        elif script_name == 'MobileNetV2_svm':
            from MobileNetV2_svm import train_mobilenetv2_svm
            train_mobilenetv2_svm()
        elif script_name == 'InceptionV3_svm':
            from InceptionV3_svm import train_inceptionv3_svm
            train_inceptionv3_svm()
        elif script_name == 'DenseNet121_svm':
            from DenseNet121_svm import train_densenet121_svm
            train_densenet121_svm()
        elif script_name == 'cnn_softmax':
            from cnn_softmax import train_cnn_softmax
            train_cnn_softmax()
        
        print(f"\n✅ Training completed for {model_name}!")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


def train_all_models():
    """Train all models sequentially"""
    print("\n🚀 TRAINING ALL MODELS")
    print("="*80)
    print("This will train all 6 models sequentially.")
    print("This process may take several hours depending on your hardware.\n")
    
    response = input("Do you want to continue? (y/n): ").lower()
    if response != 'y':
        print("Cancelled.")
        return
    
    models = [
        ('1', 'ResNet50 + SVM'),
        ('2', 'VGG16 + SVM'),
        ('3', 'MobileNetV2 + SVM'),
        ('4', 'InceptionV3 + SVM'),
        ('5', 'DenseNet121 + SVM'),
        ('6', 'CNN + Softmax')
    ]
    
    print("\n" + "="*80)
    print("Starting batch training...")
    print("="*80 + "\n")
    
    results = []
    
    for idx, (choice, name) in enumerate(models, 1):
        print(f"\n{'#'*80}")
        print(f"# MODEL {idx}/{len(models)}: {name}")
        print(f"{'#'*80}\n")
        
        try:
            train_single_model(choice)
            results.append((name, 'SUCCESS'))
        except Exception as e:
            print(f"❌ Failed to train {name}: {e}")
            results.append((name, 'FAILED'))
    
    # Summary
    print("\n" + "="*80)
    print("📊 BATCH TRAINING SUMMARY")
    print("="*80)
    
    for name, status in results:
        emoji = "✅" if status == "SUCCESS" else "❌"
        print(f"{emoji} {name}: {status}")
    
    print("="*80 + "\n")


def view_training_history():
    """View training history/reports"""
    from utils.view_reports import view_reports
    view_reports(show_per_class=True)


def compare_all_models():
    """Compare all trained models"""
    from compare_models import compare_models
    compare_models(save_comparison=True)


def test_model_menu():
    """Test a trained model"""
    print("\n📋 SELECT MODEL TO TEST:")
    print("  1. ResNet50 + SVM")
    print("  2. VGG16 + SVM")
    print("  3. MobileNetV2 + SVM")
    print("  4. InceptionV3 + SVM")
    print("  5. DenseNet121 + SVM")
    print("  6. CNN + Softmax")
    print("  0. Back to main menu")
    
    choice = input("\nEnter your choice (0-6): ").strip()
    
    if choice == '0':
        return
    
    model_map = {
        '1': 'resnet50',
        '2': 'vgg16',
        '3': 'mobilenetv2',
        '4': 'inceptionv3',
        '5': 'densenet121',
        '6': 'cnn'
    }
    
    if choice in model_map:
        from test_models import test_model
        test_model(model_map[choice])
    else:
        print("❌ Invalid choice!")


def show_menu():
    """Show main menu"""
    print("\n" + "="*80)
    print("📋 MAIN MENU")
    print("="*80)
    print("  1. Train all models")
    print("  2. Train specific model")
    print("  3. View training history / reports")
    print("  4. Compare all models")
    print("  5. Test a trained model")
    print("  6. Analyze results (detailed)")
    print("  0. Exit")
    print("="*80)


def show_model_selection_menu():
    """Show model selection menu"""
    print("\n📋 SELECT MODEL TO TRAIN:")
    print("  1. ResNet50 + SVM")
    print("  2. VGG16 + SVM")
    print("  3. MobileNetV2 + SVM")
    print("  4. InceptionV3 + SVM")
    print("  5. DenseNet121 + SVM")
    print("  6. CNN + Softmax")
    print("  0. Back to main menu")


def analyze_results():
    """Detailed analysis of results"""
    print("\n📊 DETAILED ANALYSIS")
    print("="*80 + "\n")
    
    reports = list_reports(sort_by='accuracy', reverse=True)
    
    if not reports:
        print("❌ No training reports found!")
        print("   Please train at least one model first.")
        return
    
    print(f"Analyzing {len(reports)} trained model(s)...\n")
    
    from utils.report_generator import load_report
    
    # Best and worst performing models
    best_report_path = reports[0]
    worst_report_path = reports[-1]
    
    best_report = load_report(best_report_path)
    worst_report = load_report(worst_report_path)
    
    print("🏆 BEST PERFORMING MODEL:")
    print(f"   Model: {best_report['model_info']['model_name']}")
    print(f"   Accuracy: {best_report['metrics']['accuracy']:.2f}%")
    print(f"   F1-Score: {best_report['metrics']['f1_score']:.2f}%")
    print(f"   Training Time: {best_report['model_info']['training_time_formatted']}\n")
    
    if len(reports) > 1:
        print("📉 LOWEST PERFORMING MODEL:")
        print(f"   Model: {worst_report['model_info']['model_name']}")
        print(f"   Accuracy: {worst_report['metrics']['accuracy']:.2f}%")
        print(f"   F1-Score: {worst_report['metrics']['f1_score']:.2f}%")
        print(f"   Training Time: {worst_report['model_info']['training_time_formatted']}\n")
        
        # Performance gap
        accuracy_gap = best_report['metrics']['accuracy'] - worst_report['metrics']['accuracy']
        print(f"📊 Performance Gap: {accuracy_gap:.2f}%\n")
    
    # Average metrics
    all_accuracies = []
    all_f1_scores = []
    
    for report_path in reports:
        report = load_report(report_path)
        all_accuracies.append(report['metrics']['accuracy'])
        all_f1_scores.append(report['metrics']['f1_score'])
    
    import numpy as np
    print("📈 AVERAGE METRICS ACROSS ALL MODELS:")
    print(f"   Average Accuracy: {np.mean(all_accuracies):.2f}%")
    print(f"   Average F1-Score: {np.mean(all_f1_scores):.2f}%")
    print(f"   Std Dev Accuracy: {np.std(all_accuracies):.2f}%")
    print(f"   Std Dev F1-Score: {np.std(all_f1_scores):.2f}%\n")
    
    print("="*80 + "\n")
    
    # Ask if user wants to see detailed per-model analysis
    response = input("View detailed per-class analysis for all models? (y/n): ").lower()
    if response == 'y':
        view_training_history()


def main():
    """Main function"""
    # Set up GPU for training - REQUIRED!
    from utils.gpu_utils import setup_gpu_for_training, print_memory_usage
    
    print_banner()
    
    # Initialize GPU configuration
    if not setup_gpu_for_training(verbose=True):
        print("\n❌ GPU initialization failed!")
        print("   Training requires NVIDIA GPU with CUDA support.")
        print("   Please check your GPU drivers and CUDA installation.")
        return
    
    # Check for existing reports
    check_existing_reports()
    
    # Main loop
    while True:
        show_menu()
        choice = input("\nEnter your choice (0-6): ").strip()
        
        if choice == '0':
            print("\n👋 Goodbye!")
            break
        
        elif choice == '1':
            train_all_models()
        
        elif choice == '2':
            show_model_selection_menu()
            model_choice = input("\nEnter your choice (0-6): ").strip()
            if model_choice != '0':
                train_single_model(model_choice)
        
        elif choice == '3':
            view_training_history()
        
        elif choice == '4':
            compare_all_models()
        
        elif choice == '5':
            test_model_menu()
        
        elif choice == '6':
            analyze_results()
        
        else:
            print("❌ Invalid choice! Please try again.")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
