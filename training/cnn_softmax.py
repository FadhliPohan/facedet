#!/usr/bin/env python3
"""
Custom CNN + Softmax Training Script - GPU Optimized
Uses pre-split train/val/test data with memory-efficient tf.data pipeline
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import time
import numpy as np
from pathlib import Path
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from config import MODEL_DIR, REPORT_DIR, HISTORY_DIR, CHECKPOINT_DIR, SPLIT_DATA_DIR, BATCH_SIZE, EPOCHS, EARLY_STOPPING_PATIENCE, REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR, IMG_SIZE
from utils.gpu_utils import setup_gpu_for_training, print_memory_usage
from utils.data_pipeline import create_datasets_for_training
from utils.model_builder import build_cnn_model
from utils.metrics_calculator import calculate_metrics, plot_confusion_matrix, plot_training_history
from utils.report_generator import generate_report, save_report, get_latest_report, load_report


def train_cnn_softmax():
    """Train Custom CNN + Softmax model with GPU optimization"""
    
    print("\n" + "="*80)
    print("🚀 TRAINING: Custom CNN + Softmax (GPU Optimized)")
    print("="*80)
    
    # Check if already trained
    final_model_path = MODEL_DIR / "cnn_softmax_final.h5"
    
    if final_model_path.exists():
        print(f"\n⚠️  Found existing trained model: {final_model_path}")
        
        latest_report_path = get_latest_report('CNN_Softmax')
        if latest_report_path:
            report = load_report(latest_report_path)
            print(f"\n   Previous performance:")
            print(f"   Accuracy: {report['metrics']['accuracy']:.2f}%")
            print(f"   F1-Score: {report['metrics']['f1_score']:.2f}%")
        
        print("\n❓ Do you want to retrain from scratch?")
        response = input("   This will overwrite existing model (y/n): ").lower()
        if response != 'y': return
        
        print("\n🔄 Retraining model from scratch...")
    
    print()
    start_time = time.time()
    print_memory_usage()
    
    # 1. Create datasets - ENABLE AUGMENTATION for training!
    print("Step 1: Loading pre-split datasets (with augmentation)...")
    train_ds, val_ds, test_ds, class_names, dataset_info = create_datasets_for_training(
        split_dir_path=str(SPLIT_DATA_DIR),
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        preprocessing='standard',  # CNN uses standard normalization
        augment_train=True
    )
    
    num_classes = len(class_names)
    
    # 2. Build CNN model
    print("\nStep 2: Building CNN model...")
    # Use strategy scope for potential multi-GPU support later
    strategy = tf.distribute.MirroredStrategy() if len(tf.config.list_physical_devices('GPU')) > 1 else tf.distribute.get_strategy()
    
    with strategy.scope():
        model = build_cnn_model(
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
            num_classes=num_classes
        )
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    model.summary()
    
    # 3. Setup callbacks
    print("\nStep 3: Setting up training callbacks...")
    model_checkpoint_path = CHECKPOINT_DIR / "cnn_softmax_best.keras"  # Use .keras format
    csv_logger_path = HISTORY_DIR / f"cnn_training_log_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    
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
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-6,
            verbose=1
        ),
        CSVLogger(str(csv_logger_path))
    ]
    
    # 4. Train model
    print(f"\nStep 4: Training model...")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Steps per epoch: {dataset_info['train_count'] // BATCH_SIZE}")
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n✅ Training completed\n")
    
    # 5. Save final model
    model.save(final_model_path)
    print(f"✅ Final model saved to: {final_model_path}\n")
    
    # 6. Save label info
    label_info = {'class_names': class_names, 'num_classes': len(class_names)}
    with open(MODEL_DIR / "cnn_label_info.pkl", 'wb') as f:
        pickle.dump(label_info, f)
    
    # 7. Plot training history
    history_plot_path = HISTORY_DIR / f"cnn_history_{time.strftime('%Y%m%d_%H%M%S')}.png"
    plot_training_history(history, save_path=history_plot_path)
    
    # 8. Evaluate model
    print("Step 5: Evaluating model...")
    
    # Get true labels from validation/test set
    # Note: iterating a shuffled dataset won't match predictions if not careful
    # We use val_ds which is NOT augmented and NOT shuffled in our pipeline
    
    # Collect all labels and predictions
    print("   Collecting predictions...")
    y_true = []
    y_pred = []
    
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(preds.argmax(axis=1))
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, class_names=class_names)
    
    # 9. Plot confusion matrix
    cm_path = REPORT_DIR / f"cnn_softmax_confusion_matrix_{time.strftime('%Y%m%d_%H%M%S')}.png"
    plot_confusion_matrix(metrics['confusion_matrix'], class_names, save_path=cm_path, figsize=(14, 12))
    
    # 10. Generate report
    training_time = time.time() - start_time
    from utils.gpu_utils import get_gpu_info
    gpu_info = get_gpu_info()
    
    # Get training history metrics
    final_train_acc = history.history['accuracy'][-1] * 100
    final_val_acc = history.history['val_accuracy'][-1] * 100
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    print("Step 6: Generating report...")
    report = generate_report(
        model_name='CNN_Softmax',
        model_type='Custom CNN + Softmax',
        metrics=metrics,
        training_time=training_time,
        num_samples_train=dataset_info['train_count'],
        num_samples_val=dataset_info['val_count'],
        class_names=class_names,
        hyperparameters={
            'epochs': EPOCHS,
            'actual_epochs': len(history.history['loss']),
            'batch_size': BATCH_SIZE,
            'optimizer': 'adam',
            'loss': 'sparse_categorical_crossentropy',
            'early_stopping_patience': EARLY_STOPPING_PATIENCE,
            'input_shape': f'{IMG_SIZE + (3,)}',
            'final_train_accuracy': f'{final_train_acc:.2f}%',
            'final_val_accuracy': f'{final_val_acc:.2f}%',
            'final_train_loss': f'{final_train_loss:.4f}',
            'final_val_loss': f'{final_val_loss:.4f}',
            'mixed_precision': 'FP16 (enabled)',
            'gpu_name': gpu_info['devices'][0].get('gpu_name', 'Unknown') if gpu_info['available'] else 'N/A'
        },
        notes='Custom CNN architecture training with tf.data pipeline'
    )
    
    # Add history image
    report['training_history_image'] = str(history_plot_path)
    
    report_path = save_report(report, 'CNN_Softmax', confusion_matrix_path=cm_path)
    
    # Summary
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\n📊 Final Results:")
    print(f"   Validation Accuracy:  {metrics['accuracy']:.2f}%")
    print(f"   F1-Score:  {metrics['f1_score']:.2f}%")
    print(f"\n📈 Training History:")
    print(f"   Final Train Accuracy: {final_train_acc:.2f}%")
    print(f"   Final Val Accuracy:   {final_val_acc:.2f}%")
    print(f"   Epochs: {len(history.history['loss'])}")
    print(f"\n⏱️  Training Time: {report['model_info']['training_time_formatted']}")
    print_memory_usage()
    return report


if __name__ == "__main__":
    if not setup_gpu_for_training(verbose=True):
        print("❌ GPU initialization failed. Exiting...")
        exit(1)
    train_cnn_softmax()
