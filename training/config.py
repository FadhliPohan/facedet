"""
Configuration file for training system
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent

# Use PRE-SPLIT data from splitting_data folder
SPLIT_DATA_DIR = BASE_DIR / "dataset" / "split_data"
TRAIN_DIR = SPLIT_DATA_DIR / "train"
VAL_DIR = SPLIT_DATA_DIR / "val"
TEST_DIR = SPLIT_DATA_DIR / "test"

# Results and reports - all saved to result_training
RESULT_DIR = BASE_DIR / "result_training"
REPORT_DIR = RESULT_DIR / "reports"
MODEL_DIR = RESULT_DIR / "models"
HISTORY_DIR = RESULT_DIR / "history"
CHECKPOINT_DIR = RESULT_DIR / "checkpoints"

# Create directories if they don't exist
RESULT_DIR.mkdir(exist_ok=True, parents=True)
REPORT_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR.mkdir(exist_ok=True, parents=True)
HISTORY_DIR.mkdir(exist_ok=True, parents=True)
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)

# Image parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 16  # Reduced for memory efficiency with large models
NUM_CLASSES = 10

# GPU optimization settings
USE_MIXED_PRECISION = True  # FP16 for RTX GPUs
GPU_MEMORY_GROWTH = True
PREFETCH_BUFFER_SIZE = 'auto'  # tf.data.AUTOTUNE

# Training parameters (no split needed - using pre-split data)
RANDOM_SEED = 42
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5

# SVM parameters
SVM_KERNEL = 'rbf'
SVM_C = 1.0
SVM_GAMMA = 'scale'
SVM_MAX_ITER = 1000

# Model configurations
MODELS = {
    'resnet50': {
        'name': 'ResNet50',
        'input_shape': (224, 224, 3),
        'preprocess': 'tf.keras.applications.resnet50.preprocess_input',
        'pooling': 'avg'
    },
    'vgg16': {
        'name': 'VGG16',
        'input_shape': (224, 224, 3),
        'preprocess': 'tf.keras.applications.vgg16.preprocess_input',
        'pooling': 'avg'
    },
    'mobilenetv2': {
        'name': 'MobileNetV2',
        'input_shape': (224, 224, 3),
        'preprocess': 'tf.keras.applications.mobilenet_v2.preprocess_input',
        'pooling': 'avg'
    },
    'inceptionv3': {
        'name': 'InceptionV3',
        'input_shape': (299, 299, 3),
        'preprocess': 'tf.keras.applications.inception_v3.preprocess_input',
        'pooling': 'avg'
    },
    'densenet121': {
        'name': 'DenseNet121',
        'input_shape': (224, 224, 3),
        'preprocess': 'tf.keras.applications.densenet.preprocess_input',
        'pooling': 'avg'
    },
    'cnn': {
        'name': 'CNN',
        'input_shape': (224, 224, 3),
        'preprocess': 'standard',  # (x / 255.0)
        'pooling': None
    }
}

# Class names (will be loaded dynamically from dataset)
CLASS_NAMES = []
