#!/usr/bin/env python3
"""
VGG16 + SVM Training Script
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

sys.path.append(str(Path(__file__).parent))

from config import MODEL_DIR, REPORT_DIR, SVM_KERNEL, SVM_C, SVM_GAMMA
from utils.data_loader import load_and_preprocess
from utils.model_builder import build_feature_extractor, extract_features
from utils.metrics_calculator import calculate_metrics, plot_confusion_matrix
from utils.report_generator import generate_report, save_report


def train_vgg16_svm():
    """Train VGG16 + SVM model"""
    
    print("\n" + "="*80)
    print("🚀 TRAINING: VGG16 + SVM")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    # 1. Load and preprocess data
    print("Step 1: Loading and preprocessing data...")
    X_train, X_val, y_train, y_val, class_names, label_encoder = load_and_preprocess(
        model_type='vgg16',
        img_size=(224, 224)
    )
    
    # 2. Build feature extractor
    print("\nStep 2: Building feature extractor...")
    feature_extractor = build_feature_extractor(
        model_type='vgg16',
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    
    # 3. Save feature extractor
    model_path = MODEL_DIR / "vgg16_feature_extractor.h5"
    feature_extractor.save(model_path)
    print(f"✅ Feature extractor saved to: {model_path}\n")
    
    # 4. Extract features
    print("Step 3: Extracting features...")
    print("  Extracting training features...")
    train_features = extract_features(feature_extractor, X_train, batch_size=32)
    print("  Extracting validation features...")
    val_features = extract_features(feature_extractor, X_val, batch_size=32)
    
    # 5. Train SVM
    print("Step 4: Training SVM classifier...")
    print(f"  Kernel: {SVM_KERNEL}")
    print(f"  C: {SVM_C}")
    print(f"  Gamma: {SVM_GAMMA}\n")
    
    svm_classifier = SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA, verbose=True)
    svm_classifier.fit(train_features, y_train)
    print("\n✅ SVM training completed\n")
    
    # 6. Save SVM classifier
    svm_path = MODEL_DIR / "vgg16_svm_classifier.pkl"
    with open(svm_path, 'wb') as f:
        pickle.dump(svm_classifier, f)
    print(f"✅ SVM classifier saved to: {svm_path}\n")
    
    # 7. Save label encoder
    encoder_path = MODEL_DIR / "vgg16_label_encoder.pkl"
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"✅ Label encoder saved to: {encoder_path}\n")
    
    # 8. Evaluate
    print("Step 5: Evaluating model...")
    y_pred = svm_classifier.predict(val_features)
    metrics = calculate_metrics(y_val, y_pred, class_names=class_names)
    
    # 9. Plot confusion matrix
    cm_path = REPORT_DIR / f"vgg16_svm_confusion_matrix_{time.strftime('%Y%m%d_%H%M%S')}.png"
    plot_confusion_matrix(metrics['confusion_matrix'], class_names, save_path=cm_path, figsize=(14, 12))
    
    # 10. Generate report
    training_time = time.time() - start_time
    print("Step 6: Generating report...")
    report = generate_report(
        model_name='VGG16_SVM',
        model_type='VGG16 + SVM',
        metrics=metrics,
        training_time=training_time,
        num_samples_train=len(X_train),
        num_samples_val=len(X_val),
        class_names=class_names,
        hyperparameters={
            'svm_kernel': SVM_KERNEL,
            'svm_C': SVM_C,
            'svm_gamma': SVM_GAMMA,
            'input_shape': '(224, 224, 3)',
            'feature_extractor': 'VGG16 (ImageNet)',
            'pooling': 'avg'
        },
        notes='Feature extraction using pretrained VGG16, classification with SVM'
    )
    
    report_path = save_report(report, 'VGG16_SVM', confusion_matrix_path=cm_path)
    
    # 11. Summary
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\n📊 Final Results:")
    print(f"   Accuracy:  {metrics['accuracy']:.2f}%")
    print(f"   Precision: {metrics['precision']:.2f}%")
    print(f"   Recall:    {metrics['recall']:.2f}%")
    print(f"   F1-Score:  {metrics['f1_score']:.2f}%")
    print(f"\n⏱️  Training Time: {report['model_info']['training_time_formatted']}")
    print(f"\n💾 Saved Files:")
    print(f"   Model: {model_path}")
    print(f"   SVM: {svm_path}")
    print(f"   Label Encoder: {encoder_path}")
    print(f"   Report: {report_path}")
    print(f"   Confusion Matrix: {cm_path}")
    print("\n" + "="*80 + "\n")
    
    return report


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    train_vgg16_svm()
