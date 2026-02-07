#!/usr/bin/env python3
"""
DenseNet121 + SVM Training Script - GPU Optimized
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


def train_densenet121_svm():
    """Train DenseNet121 + SVM model with GPU optimization"""
    
    print("\n" + "="*80)
    print("🚀 TRAINING: DenseNet121 + SVM (GPU Optimized)")
    print("="*80)
    
    # Check if already trained
    model_path = MODEL_DIR / "densenet121_feature_extractor.h5"
    svm_path = MODEL_DIR / "densenet121_svm_classifier.pkl"
    
    if model_path.exists() and svm_path.exists():
        print("\n⚠️  Found existing trained model:")
        print(f"   Feature extractor: {model_path}")
        print(f"   SVM classifier: {svm_path}")
        
        latest_report_path = get_latest_report('DenseNet121_SVM')
        
        if latest_report_path:
            report = load_report(latest_report_path)
            print(f"\n   Previous performance:")
            print(f"   Accuracy: {report['metrics']['accuracy']:.2f}%")
            print(f"   F1-Score: {report['metrics']['f1_score']:.2f}%")
        
        print("\n❓ Do you want to retrain from scratch?")
        response = input("   This will overwrite existing model (y/n): ").lower()
        if response != 'y': return
        
        print("\n🔄 Retraining model from scratch...")
        clear_gpu_memory()
    
    print()
    start_time = time.time()
    print_memory_usage()
    
    # 1. Create datasets
    print("Step 1: Loading pre-split datasets...")
    train_ds, val_ds, test_ds, class_names, dataset_info = create_datasets_for_training(
        split_dir_path=str(SPLIT_DATA_DIR),
        img_size=(224, 224),
        batch_size=BATCH_SIZE,
        preprocessing='densenet121',
        augment_train=False
    )
    
    # 2. Build feature extractor
    print("Step 2: Building DenseNet121 feature extractor...")
    feature_extractor = build_feature_extractor(
        model_type='densenet121',
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    
    # 3. Save feature extractor
    feature_extractor.save(model_path)
    print(f"✅ Feature extractor saved to: {model_path}\n")
    
    # 4. Extract features
    print("Step 3: Extracting features (batch processing)...")
    
    train_features, train_labels = extract_features_from_dataset(
        feature_extractor, train_ds, dataset_info['train_count'], BATCH_SIZE
    )
    
    val_features, val_labels = extract_features_from_dataset(
        feature_extractor, val_ds, dataset_info['val_count'], BATCH_SIZE
    )
    
    # Clear GPU memory
    print("🧹 Clearing GPU memory...")
    clear_gpu_memory()
    print_memory_usage()
    
    # 5. Train SVM classifier
    print("Step 4: Training SVM classifier...")
    svm_classifier = SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA, verbose=True)
    svm_classifier.fit(train_features, train_labels)
    print("\n✅ SVM training completed\n")
    
    # 6. Save SVM classifier
    with open(svm_path, 'wb') as f:
        pickle.dump(svm_classifier, f)
    print(f"✅ SVM classifier saved to: {svm_path}\n")
    
    # 7. Save label info
    label_info = {'class_names': class_names, 'num_classes': len(class_names)}
    with open(MODEL_DIR / "densenet121_label_info.pkl", 'wb') as f:
        pickle.dump(label_info, f)
    
    # 8. Evaluate
    print("Step 5: Evaluating on validation set...")
    y_pred = svm_classifier.predict(val_features)
    metrics = calculate_metrics(val_labels, y_pred, class_names=class_names)
    
    # 9. Plot confusion matrix
    cm_path = REPORT_DIR / f"densenet121_svm_confusion_matrix_{time.strftime('%Y%m%d_%H%M%S')}.png"
    plot_confusion_matrix(metrics['confusion_matrix'], class_names, save_path=cm_path, figsize=(14, 12))
    
    # 10. Generate report
    training_time = time.time() - start_time
    from utils.gpu_utils import get_gpu_info
    gpu_info = get_gpu_info()
    
    print("Step 6: Generating report...")
    report = generate_report(
        model_name='DenseNet121_SVM',
        model_type='DenseNet121 + SVM',
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
            'feature_extractor': 'DenseNet121 (ImageNet)',
            'pooling': 'avg',
            'batch_size': BATCH_SIZE,
            'mixed_precision': 'FP16 (enabled)'
        },
        notes='Feature extraction using pretrained DenseNet121, classification with SVM'
    )
    
    report_path = save_report(report, 'DenseNet121_SVM', confusion_matrix_path=cm_path)
    
    # Summary
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\n📊 Final Results (Validation Set):")
    print(f"   Accuracy:  {metrics['accuracy']:.2f}%")
    print(f"   F1-Score:  {metrics['f1_score']:.2f}%")
    print(f"\n⏱️  Training Time: {report['model_info']['training_time_formatted']}")
    print_memory_usage()
    return report


if __name__ == "__main__":
    if not setup_gpu_for_training(verbose=True):
        print("❌ GPU initialization failed. Exiting...")
        exit(1)
    train_densenet121_svm()
