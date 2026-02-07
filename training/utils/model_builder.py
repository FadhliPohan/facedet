"""
Model building utilities
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    ResNet50, VGG16, MobileNetV2, InceptionV3, DenseNet121
)
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import IMG_SIZE, NUM_CLASSES


def build_feature_extractor(model_type='resnet50', input_shape=(224, 224, 3), pooling='avg'):
    """
    Build feature extractor from pretrained models
    
    Args:
        model_type: Type of model ('resnet50', 'vgg16', 'mobilenetv2', 'inceptionv3', 'densenet121')
        input_shape: Input image shape
        pooling: Pooling type ('avg' or 'max')
        
    Returns:
        Feature extractor model
    """
    print(f"\n{'='*80}")
    print(f"🔨 Building {model_type.upper()} feature extractor")
    print(f"{'='*80}\n")
    
    if model_type == 'resnet50':
        base_model = ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
            pooling=pooling
        )
    elif model_type == 'vgg16':
        base_model = VGG16(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
            pooling=pooling
        )
    elif model_type == 'mobilenetv2':
        base_model = MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
            pooling=pooling
        )
    elif model_type == 'inceptionv3':
        base_model = InceptionV3(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
            pooling=pooling
        )
    elif model_type == 'densenet121':
        base_model = DenseNet121(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
            pooling=pooling
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Freeze base model layers
    base_model.trainable = False
    
    print(f"✅ Feature extractor built successfully")
    print(f"   Total parameters: {base_model.count_params():,}")
    print(f"   Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in base_model.trainable_weights]):,}")
    print(f"   Output shape: {base_model.output_shape}")
    print()
    
    return base_model


def build_cnn_model(input_shape=(224, 224, 3), num_classes=NUM_CLASSES):
    """
    Build custom CNN model with Softmax classifier
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        
    Returns:
        CNN model
    """
    print(f"\n{'='*80}")
    print(f"🔨 Building Custom CNN model")
    print(f"{'='*80}\n")
    
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # Conv Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Conv Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Conv Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Conv Block 4
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and Dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✅ CNN model built successfully")
    print(f"   Total parameters: {model.count_params():,}")
    print(f"   Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
    print()
    
    return model


def extract_features(model, X, batch_size=32, verbose=True):
    """
    Extract features from images using the model
    
    Args:
        model: Feature extractor model
        X: Input images
        batch_size: Batch size for prediction
        verbose: Show progress
        
    Returns:
        Extracted features
    """
    if verbose:
        print(f"Extracting features from {len(X):,} images...")
    
    features = model.predict(X, batch_size=batch_size, verbose=1 if verbose else 0)
    
    if verbose:
        print(f"✅ Features extracted: {features.shape}")
        print()
    
    return features
