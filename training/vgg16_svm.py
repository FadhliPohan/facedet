#!/usr/bin/env python3
"""
VGG16 + SVM Training Script - GPU Optimized
Uses pre-split train/val/test data with memory-efficient tf.data pipeline
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import time
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
import pickle
import tensorflow as tf

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from config import MODEL_DIR, REPORT_DIR, SVM_KERNEL, SVM_C, SVM_GAMMA, SPLIT_DATA_DIR, BATCH_SIZE
from utils.gpu_utils import setup_gpu_for_training, clear_gpu_memory, print_memory_usage
from utils.data_pipeline import create_datasets_for_training, extract_features_from_dataset, get_dataset_info
from utils.model_builder import build_feature_extractor
from utils.metrics_calculator import calculate_metrics, plot_confusion_matrix
from utils.report_generator import generate_report, save_report, get_latest_report, load_report


def train_vgg16_svm():
    """Train VGG16 + SVM model with GPU optimization"""
    
    print("\n" + "="*80)
    print("🚀 TRAINING: VGG16 + SVM (GPU Optimized)")
    print("="*80)
    
    # Check if already trained
    model_path = MODEL_DIR / "vgg16_feature_extractor.h5"
    svm_path = MODEL_DIR / "vgg16_svm_classifier.pkl"
    
    if model_path.exists() and svm_path.exists():
        print("\n⚠️  Found existing trained model:")
        print(f"   Feature extractor: {model_path}")
        print(f"   SVM classifier: {svm_path}")
        
        # Try to load latest report to show metrics
        latest_report_path = get_latest_report('VGG16_SVM')
        
        if latest_report_path:
            report = load_report(latest_report_path)
            print(f"\n   Previous performance:")
            print(f"   Accuracy: {report['metrics']['accuracy']:.2f}%")
            print(f"   F1-Score: {report['metrics']['f1_score']:.2f}%")
            print(f"   Trained on: {report['model_info']['timestamp']}")
        
        print("\n❓ Do you want to retrain from scratch?")
        response = input("   This will overwrite existing model (y/n): ").lower()
        
        if response != 'y':
            print("\n✅ Skipping training - using existing model")
            return
        
        print("\n🔄 Retraining model from scratch...")
        clear_gpu_memory()
    
    print()
    start_time = time.time()
    
    # Show memory before starting
    print_memory_usage()
    
    # 1. Create datasets
    print("Step 1: Loading pre-split datasets...")
    train_ds, val_ds, test_ds, class_names, dataset_info = create_datasets_for_training(
        split_dir_path=str(SPLIT_DATA_DIR),
        img_size=(224, 224),
        batch_size=BATCH_SIZE,
        preprocessing='vgg16',
        augment_train=False
    )
    
    # 2. Build feature extractor
    print("Step 2: Building VGG16 feature extractor...")
    feature_extractor = build_feature_extractor(
        model_type='vgg16',
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    
    # 3. Save feature extractor
    feature_extractor.save(model_path)
    print(f"✅ Feature extractor saved to: {model_path}\n")
    
    # 4. Extract features
    print("Step 3: Extracting features (batch processing)...")
    
    train_features, train_labels = extract_features_from_dataset(
        feature_extractor,
        train_ds,
        total_samples=dataset_info['train_count'],
        batch_size=BATCH_SIZE,
        desc="Extracting training features"
    )
    
    val_features, val_labels = extract_features_from_dataset(
        feature_extractor,
        val_ds,
        total_samples=dataset_info['val_count'],
        batch_size=BATCH_SIZE,
        desc="Extracting validation features"
    )
    
    # Clear GPU memory
    print("🧹 Clearing GPU memory...")
    clear_gpu_memory()
    print_memory_usage()
    
    # 5. Train SVM classifier
    print("Step 4: Training SVM classifier...")
    print(f"  Kernel: {SVM_KERNEL}")
    print(f"  C: {SVM_C}")
    print(f"  Gamma: {SVM_GAMMA}")
    print(f"  Training samples: {len(train_features):,}")
    print(f"  Feature dimension: {train_features.shape[1]}\n")
    
    svm_classifier = SVC(
        kernel=SVM_KERNEL,
        C=SVM_C,
        gamma=SVM_GAMMA,
        verbose=True
    )
    
    svm_classifier.fit(train_features, train_labels)
    print("\n✅ SVM training completed\n")
    
    # 6. Save SVM classifier
    with open(svm_path, 'wb') as f:
        pickle.dump(svm_classifier, f)
    print(f"✅ SVM classifier saved to: {svm_path}\n")
    
    # 7. Save label info
    label_info = {
        'class_names': class_names,
        'num_classes': len(class_names)
    }
    encoder_path = MODEL_DIR / "vgg16_label_info.pkl"
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_info, f)
    print(f"✅ Label info saved to: {encoder_path}\n")
    
    # 8. Evaluate on validation set
    print("Step 5: Evaluating on validation set...")
    y_pred = svm_classifier.predict(val_features)
    
    # 9. Calculate metrics
    metrics = calculate_metrics(val_labels, y_pred, class_names=class_names)
    
    # 10. Plot confusion matrix
    cm_path = REPORT_DIR / f"vgg16_svm_confusion_matrix_{time.strftime('%Y%m%d_%H%M%S')}.png"
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        class_names,
        save_path=cm_path,
        figsize=(14, 12)
    )
    
    # 11. Calculate training time
    training_time = time.time() - start_time
    
    # 12. Get GPU info
    from utils.gpu_utils import get_gpu_info
    gpu_info = get_gpu_info()
    
    # 13. Generate report
    print("Step 6: Generating report...")
    report = generate_report(
        model_name='VGG16_SVM',
        model_type='VGG16 + SVM',
        metrics=metrics,
        training_time=training_time,
        num_samples_train=dataset_info['train_count'],
        num_samples_val=dataset_info['val_count'],
        class_names=class_names,
        hyperparameters={
            'svm_kernel': SVM_KERNEL,
            'svm_C': SVM_C,
            'svm_gamma': SVM_GAMMA,
            'input_shape': '(224, 224, 3)',
            'feature_extractor': 'VGG16 (ImageNet)',
            'pooling': 'avg',
            'batch_size': BATCH_SIZE,
            'mixed_precision': 'FP16 (enabled)',
            'gpu_name': gpu_info['devices'][0].get('gpu_name', 'Unknown') if gpu_info['available'] else 'N/A'
        },
        notes='Feature extraction using pretrained VGG16, classification with SVM'
    )
    
    report_path = save_report(report, 'VGG16_SVM', confusion_matrix_path=cm_path)
    
    # 14. Final summary
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\n📊 Final Results (Validation Set):")
    print(f"   Accuracy:  {metrics['accuracy']:.2f}%")
    print(f"   F1-Score:  {metrics['f1_score']:.2f}%")
    print(f"\n⏱️  Training Time: {report['model_info']['training_time_formatted']}")
    print(f"\n💾 Saved Files:")
    print(f"   Feature Extractor: {model_path}")
    print(f"   SVM Classifier: {svm_path}")
    print(f"   Report: {report_path}")
    print("\n" + "="*80 + "\n")
    
    print_memory_usage()
    return report


if __name__ == "__main__":
    if not setup_gpu_for_training(verbose=True):
        print("❌ GPU initialization failed. Exiting...")
        exit(1)
    train_vgg16_svm()
