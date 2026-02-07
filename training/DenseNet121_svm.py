#!/usr/bin/env python3
"""
DenseNet121 + SVM Training Script
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import time
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


def train_densenet121_svm():
    """Train DenseNet121 + SVM model"""
    
    print("\n" + "="*80)
    print("🚀 TRAINING: DenseNet121 + SVM")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    print("Step 1: Loading and preprocessing data...")
    X_train, X_val, y_train, y_val, class_names, label_encoder = load_and_preprocess(
        model_type='densenet121',
        img_size=(224, 224)
    )
    
    print("\nStep 2: Building feature extractor...")
    feature_extractor = build_feature_extractor(
        model_type='densenet121',
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    
    model_path = MODEL_DIR / "densenet121_feature_extractor.h5"
    feature_extractor.save(model_path)
    print(f"✅ Feature extractor saved to: {model_path}\n")
    
    print("Step 3: Extracting features...")
    print("  Extracting training features...")
    train_features = extract_features(feature_extractor, X_train, batch_size=32)
    print("  Extracting validation features...")
    val_features = extract_features(feature_extractor, X_val, batch_size=32)
    
    print("Step 4: Training SVM classifier...")
    print(f"  Kernel: {SVM_KERNEL}, C: {SVM_C}, Gamma: {SVM_GAMMA}\n")
    
    svm_classifier = SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA, verbose=True)
    svm_classifier.fit(train_features, y_train)
    print("\n✅ SVM training completed\n")
    
    svm_path = MODEL_DIR / "densenet121_svm_classifier.pkl"
    with open(svm_path, 'wb') as f:
        pickle.dump(svm_classifier, f)
    print(f"✅ SVM classifier saved to: {svm_path}\n")
    
    encoder_path = MODEL_DIR / "densenet121_label_encoder.pkl"
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"✅ Label encoder saved to: {encoder_path}\n")
    
    print("Step 5: Evaluating model...")
    y_pred = svm_classifier.predict(val_features)
    metrics = calculate_metrics(y_val, y_pred, class_names=class_names)
    
    cm_path = REPORT_DIR / f"densenet121_svm_confusion_matrix_{time.strftime('%Y%m%d_%H%M%S')}.png"
    plot_confusion_matrix(metrics['confusion_matrix'], class_names, save_path=cm_path, figsize=(14, 12))
    
    training_time = time.time() - start_time
    print("Step 6: Generating report...")
    report = generate_report(
        model_name='DenseNet121_SVM',
        model_type='DenseNet121 + SVM',
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
            'feature_extractor': 'DenseNet121 (ImageNet)',
            'pooling': 'avg'
        },
        notes='Feature extraction using pretrained DenseNet121, classification with SVM'
    )
    
    report_path = save_report(report, 'DenseNet121_SVM', confusion_matrix_path=cm_path)
    
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
    
    train_densenet121_svm()
