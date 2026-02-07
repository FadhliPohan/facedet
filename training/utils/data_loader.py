"""
Data loading and preprocessing utilities
"""
import os
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import sys

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent))
from config import DATASET_DIR, IMG_SIZE, TRAIN_SPLIT, RANDOM_SEED


def load_dataset(img_size=IMG_SIZE, max_samples_per_class=None, verbose=True):
    """
    Load dataset from final_images folder
    
    Args:
        img_size: Target image size (height, width)
        max_samples_per_class: Maximum samples to load per class (for testing)
        verbose: Print progress information
        
    Returns:
        X: Image data array
        y: Labels
        class_names: List of class names
        label_encoder: Fitted LabelEncoder object
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"📂 Loading dataset from: {DATASET_DIR}")
        print(f"{'='*80}\n")
    
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")
    
    # Get class folders
    class_folders = sorted([d for d in DATASET_DIR.iterdir() if d.is_dir()])
    class_names = [folder.name for folder in class_folders]
    
    if verbose:
        print(f"Found {len(class_names)} classes:")
        for i, name in enumerate(class_names, 1):
            print(f"  {i}. {name}")
        print()
    
    # Load images
    X = []
    y = []
    
    for class_idx, class_folder in enumerate(class_folders):
        class_name = class_folder.name
        image_files = list(class_folder.glob('*.jpg')) + \
                     list(class_folder.glob('*.jpeg')) + \
                     list(class_folder.glob('*.png'))
        
        # Limit samples if specified
        if max_samples_per_class:
            image_files = image_files[:max_samples_per_class]
        
        if verbose:
            desc = f"Loading {class_name[:40]:40s}"
            pbar = tqdm(image_files, desc=desc, ncols=100)
        else:
            pbar = image_files
        
        for img_path in pbar:
            try:
                # Read image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize
                img = cv2.resize(img, img_size)
                
                X.append(img)
                y.append(class_name)
                
            except Exception as e:
                if verbose:
                    print(f"Warning: Failed to load {img_path}: {e}")
                continue
    
    # Convert to numpy arrays
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"✅ Dataset loaded successfully!")
        print(f"   Total images: {len(X):,}")
        print(f"   Image shape: {X.shape[1:]}")
        print(f"   Number of classes: {len(class_names)}")
        print(f"{'='*80}\n")
        
        # Print class distribution
        print("Class distribution:")
        unique, counts = np.unique(y_encoded, return_counts=True)
        for class_idx, count in zip(unique, counts):
            print(f"  {label_encoder.classes_[class_idx][:40]:40s}: {count:,} images")
        print()
    
    return X, y_encoded, class_names, label_encoder


def preprocess_data(X, y, preprocess_type='standard', test_size=0.2, random_state=RANDOM_SEED):
    """
    Preprocess data and split into train/validation sets
    
    Args:
        X: Image data
        y: Labels
        preprocess_type: 'standard' (divide by 255) or specific model preprocessing
        test_size: Validation split ratio
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_val, y_train, y_val
    """
    # Normalization
    if preprocess_type == 'standard':
        X = X / 255.0
    elif preprocess_type == 'resnet50':
        from tensorflow.keras.applications.resnet50 import preprocess_input
        X = preprocess_input(X)
    elif preprocess_type == 'vgg16':
        from tensorflow.keras.applications.vgg16 import preprocess_input
        X = preprocess_input(X)
    elif preprocess_type == 'mobilenetv2':
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        X = preprocess_input(X)
    elif preprocess_type == 'inceptionv3':
        from tensorflow.keras.applications.inception_v3 import preprocess_input
        X = preprocess_input(X)
    elif preprocess_type == 'densenet121':
        from tensorflow.keras.applications.densenet import preprocess_input
        X = preprocess_input(X)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Data split:")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Validation samples: {len(X_val):,}")
    print()
    
    return X_train, X_val, y_train, y_val


def load_and_preprocess(model_type='resnet50', img_size=IMG_SIZE, max_samples=None):
    """
    Convenience function to load and preprocess data in one call
    
    Args:
        model_type: Type of model (determines preprocessing)
        img_size: Target image size
        max_samples: Maximum samples per class (for testing)
        
    Returns:
        X_train, X_val, y_train, y_val, class_names, label_encoder
    """
    # Load dataset
    X, y, class_names, label_encoder = load_dataset(img_size=img_size, max_samples_per_class=max_samples)
    
    # Determine preprocessing type
    if model_type in ['resnet50', 'vgg16', 'mobilenetv2', 'inceptionv3', 'densenet121']:
        preprocess_type = model_type
    else:
        preprocess_type = 'standard'
    
    # Preprocess and split
    X_train, X_val, y_train, y_val = preprocess_data(X, y, preprocess_type=preprocess_type)
    
    return X_train, X_val, y_train, y_val, class_names, label_encoder
