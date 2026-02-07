#!/usr/bin/env python3
"""
Custom CNN + Softmax Training Script
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import time
from pathlib import Path
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

sys.path.append(str(Path(__file__).parent))

from config import MODEL_DIR, REPORT_DIR, EPOCHS, EARLY_STOPPING_PATIENCE
from utils.data_loader import load_and_preprocess
from utils.model_builder import build_cnn_model
from utils.metrics_calculator import calculate_metrics, plot_confusion_matrix, plot_training_history
from utils.report_generator import generate_report, save_report


def train_cnn_softmax():
    """Train Custom CNN + Softmax model"""
    
    print("\n" + "="*80)
    print("🚀 TRAINING: Custom CNN + Softmax")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    # 1. Load and preprocess data
    print("Step 1: Loading and preprocessing data...")
    X_train, X_val, y_train, y_val, class_names, label_encoder = load_and_preprocess(
        model_type='cnn',  # standard normalization
        img_size=(224, 224)
    )
    
    num_classes = len(class_names)
    
    # 2. Build CNN model
    print("\nStep 2: Building CNN model...")
    model = build_cnn_model(
        input_shape=(224, 224, 3),
        num_classes=num_classes
    )
    
    # Print model summary
    model.summary()
    
    # 3. Setup callbacks
    print("\nStep 3: Setting up training callbacks...")
    model_checkpoint_path = MODEL_DIR / "cnn_softmax_best.h5"
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            str(model_checkpoint_path),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # 4. Train model
    print(f"\nStep 4: Training model...")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print(f"  Batch size: 32\n")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n✅ Training completed\n")
    
    # 5. Save final model
    final_model_path = MODEL_DIR / "cnn_softmax_final.h5"
    model.save(final_model_path)
    print(f"✅ Final model saved to: {final_model_path}\n")
    
    # 6. Save label encoder
    encoder_path = MODEL_DIR / "cnn_label_encoder.pkl"
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"✅ Label encoder saved to: {encoder_path}\n")
    
    # 7. Plot training history
    history_plot_path = REPORT_DIR / f"cnn_softmax_training_history_{time.strftime('%Y%m%d_%H%M%S')}.png"
    plot_training_history(history, save_path=history_plot_path)
    
    # 8. Evaluate model
    print("Step 5: Evaluating model...")
    y_pred_probs = model.predict(X_val, verbose=0)
    y_pred = y_pred_probs.argmax(axis=1)
    
    # Calculate metrics
    metrics = calculate_metrics(y_val, y_pred, class_names=class_names)
    
    # 9. Plot confusion matrix
    cm_path = REPORT_DIR / f"cnn_softmax_confusion_matrix_{time.strftime('%Y%m%d_%H%M%S')}.png"
    plot_confusion_matrix(metrics['confusion_matrix'], class_names, save_path=cm_path, figsize=(14, 12))
    
    # 10. Calculate training time
    training_time = time.time() - start_time
    
    # 11. Generate and save report
    print("Step 6: Generating report...")
    
    # Get training history metrics
    final_train_acc = history.history['accuracy'][-1] * 100
    final_val_acc = history.history['val_accuracy'][-1] * 100
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    report = generate_report(
        model_name='CNN_Softmax',
        model_type='Custom CNN + Softmax',
        metrics=metrics,
        training_time=training_time,
        num_samples_train=len(X_train),
        num_samples_val=len(X_val),
        class_names=class_names,
        hyperparameters={
            'epochs': EPOCHS,
            'actual_epochs': len(history.history['loss']),
            'batch_size': 32,
            'optimizer': 'adam',
            'loss': 'sparse_categorical_crossentropy',
            'early_stopping_patience': EARLY_STOPPING_PATIENCE,
            'input_shape': '(224, 224, 3)',
            'final_train_accuracy': f'{final_train_acc:.2f}%',
            'final_val_accuracy': f'{final_val_acc:.2f}%',
            'final_train_loss': f'{final_train_loss:.4f}',
            'final_val_loss': f'{final_val_loss:.4f}'
        },
        notes='Custom CNN architecture trained from scratch with Softmax classifier'
    )
    
    # Add training history image to report
    report['training_history_image'] = str(history_plot_path)
    
    report_path = save_report(report, 'CNN_Softmax', confusion_matrix_path=cm_path)
    
    # 12. Summary
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\n📊 Final Results:")
    print(f"   Validation Accuracy:  {metrics['accuracy']:.2f}%")
    print(f"   Precision: {metrics['precision']:.2f}%")
    print(f"   Recall:    {metrics['recall']:.2f}%")
    print(f"   F1-Score:  {metrics['f1_score']:.2f}%")
    print(f"\n📈 Training History:")
    print(f"   Final Train Accuracy: {final_train_acc:.2f}%")
    print(f"   Final Val Accuracy:   {final_val_acc:.2f}%")
    print(f"   Epochs Trained: {len(history.history['loss'])}/{EPOCHS}")
    print(f"\n⏱️  Training Time: {report['model_info']['training_time_formatted']}")
    print(f"\n💾 Saved Files:")
    print(f"   Best Model: {model_checkpoint_path}")
    print(f"   Final Model: {final_model_path}")
    print(f"   Label Encoder: {encoder_path}")
    print(f"   Report: {report_path}")
    print(f"   Confusion Matrix: {cm_path}")
    print(f"   Training History: {history_plot_path}")
    print("\n" + "="*80 + "\n")
    
    return report


if __name__ == "__main__":
    # Set GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    train_cnn_softmax()
