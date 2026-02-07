#!/usr/bin/env python3
"""
Test Trained Models
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import pickle
from pathlib import Path
import numpy as np
import tensorflow as tf

sys.path.append(str(Path(__file__).parent))

from config import MODEL_DIR, REPORT_DIR
from utils.data_loader import load_dataset, preprocess_data
from utils.model_builder import extract_features
from utils.metrics_calculator import calculate_metrics, plot_confusion_matrix
from utils.report_generator import generate_report, save_report
import time


def test_model(model_name, test_data_path=None):
    """
    Test a trained model
    
    Args:
        model_name: Name of model to test ('resnet50', 'vgg16', 'mobilenetv2', 
                    'inceptionv3', 'densenet121', 'cnn')
        test_data_path: Path to test data (if None, uses validation split)
    """
    print("\n" + "="*80)
    print(f"🧪 TESTING: {model_name.upper()}")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    # Determine image size
    img_size = (299, 299) if model_name == 'inceptionv3' else (224, 224)
    
    # Load data
    print("Loading test data...")
    if test_data_path:
        # Load custom test data
        # TODO: Implement custom test data loading
        print("Custom test data loading not yet implemented")
        return
    else:
        # Use validation split from dataset
        X, y, class_names, label_encoder = load_dataset(img_size=img_size)
        
        # Use validation split
        from sklearn.model_selection import train_test_split
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    
    print(f"Test samples: {len(X_test):,}\n")
    
    # Load model and make predictions
    if model_name in ['resnet50', 'vgg16', 'mobilenetv2', 'inceptionv3', 'densenet121']:
        # SVM-based models
        print("Loading feature extractor and SVM classifier...")
        
        # Load feature extractor
        feature_extractor_path = MODEL_DIR / f"{model_name}_feature_extractor.h5"
        if not feature_extractor_path.exists():
            print(f"❌ Model not found: {feature_extractor_path}")
            print("   Please train the model first.")
            return
        
        feature_extractor = tf.keras.models.load_model(feature_extractor_path)
        
        # Load SVM classifier
        svm_path = MODEL_DIR / f"{model_name}_svm_classifier.pkl"
        with open(svm_path, 'rb') as f:
            svm_classifier = pickle.load(f)
        
        # Load label encoder
        encoder_path = MODEL_DIR / f"{model_name}_label_encoder.pkl"
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        
        # Preprocess data
        if model_name == 'resnet50':
            from tensorflow.keras.applications.resnet50 import preprocess_input
        elif model_name == 'vgg16':
            from tensorflow.keras.applications.vgg16 import preprocess_input
        elif model_name == 'mobilenetv2':
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        elif model_name == 'inceptionv3':
            from tensorflow.keras.applications.inception_v3 import preprocess_input
        elif model_name == 'densenet121':
            from tensorflow.keras.applications.densenet import preprocess_input
        
        X_test_preprocessed = preprocess_input(X_test.astype(np.float32))
        
        # Extract features
        print("Extracting features...")
        test_features = extract_features(feature_extractor, X_test_preprocessed, batch_size=32)
        
        # Predict
        print("Making predictions...")
        y_pred = svm_classifier.predict(test_features)
        
    elif model_name == 'cnn':
        # CNN model
        print("Loading CNN model...")
        
        model_path = MODEL_DIR / "cnn_softmax_best.h5"
        if not model_path.exists():
            model_path = MODEL_DIR / "cnn_softmax_final.h5"
        
        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            print("   Please train the model first.")
            return
        
        model = tf.keras.models.load_model(model_path)
        
        # Load label encoder
        encoder_path = MODEL_DIR / "cnn_label_encoder.pkl"
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        
        # Preprocess (standard normalization)
        X_test_preprocessed = X_test / 255.0
        
        # Predict
        print("Making predictions...")
        y_pred_probs = model.predict(X_test_preprocessed, batch_size=32, verbose=1)
        y_pred = y_pred_probs.argmax(axis=1)
    
    else:
        print(f"❌ Unknown model name: {model_name}")
        return
    
    # Calculate metrics
    print("\nCalculating metrics...")
    class_names_list = list(label_encoder.classes_)
    metrics = calculate_metrics(y_test, y_pred, class_names=class_names_list)
    
    # Plot confusion matrix
    cm_path = REPORT_DIR / f"{model_name}_test_confusion_matrix_{time.strftime('%Y%m%d_%H%M%S')}.png"
    plot_confusion_matrix(metrics['confusion_matrix'], class_names_list, save_path=cm_path, figsize=(14, 12))
    
    # Generate test report
    test_time = time.time() - start_time
    
    report = generate_report(
        model_name=f'{model_name.upper()}_TEST',
        model_type=f'{model_name.upper()} Test Evaluation',
        metrics=metrics,
        training_time=test_time,
        num_samples_train=0,
        num_samples_val=len(X_test),
        class_names=class_names_list,
        hyperparameters={},
        notes='Test evaluation on validation set'
    )
    
    report_path = save_report(report, f'{model_name}_TEST', confusion_matrix_path=cm_path)
    
    # Summary
    print("\n" + "="*80)
    print("✅ TESTING COMPLETED!")
    print("="*80)
    print(f"\n📊 Test Results:")
    print(f"   Accuracy:  {metrics['accuracy']:.2f}%")
    print(f"   Precision: {metrics['precision']:.2f}%")
    print(f"   Recall:    {metrics['recall']:.2f}%")
    print(f"   F1-Score:  {metrics['f1_score']:.2f}%")
    print(f"\n⏱️  Test Time: {report['model_info']['training_time_formatted']}")
    print(f"\n💾 Saved Files:")
    print(f"   Report: {report_path}")
    print(f"   Confusion Matrix: {cm_path}")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test trained models')
    parser.add_argument('model', type=str, 
                       choices=['resnet50', 'vgg16', 'mobilenetv2', 'inceptionv3', 'densenet121', 'cnn'],
                       help='Model to test')
    
    args = parser.parse_args()
    
    # Set GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    test_model(args.model)
