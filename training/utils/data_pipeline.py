"""
Optimized Data Pipeline using tf.data.Dataset
Memory-efficient loading from pre-split train/val/test folders
"""
import tensorflow as tf
import numpy as np
from pathlib import Path
import json
import sys

# Add parent directory for config import
sys.path.append(str(Path(__file__).parent.parent))


def get_dataset_info(split_dir_path):
    """
    Get dataset information without loading images
    
    Args:
        split_dir_path: Path to splitting_data folder
        
    Returns:
        dict with dataset info
    """
    split_dir = Path(split_dir_path)
    
    # Read from split_report.json if available
    report_file = split_dir / 'split_report.json'
    
    if report_file.exists():
        with open(report_file, 'r') as f:
            report = json.load(f)
        
        return {
            'num_classes': report['num_classes'],
            'class_names': report['class_names'],
            'train_count': report['train']['count'],
            'val_count': report['val']['count'],
            'test_count': report['test']['count'],
            'total_count': report['total_images']
        }
    
    # Fallback: count manually
    class_folders = sorted([d for d in (split_dir / 'train').iterdir() if d.is_dir()])
    class_names = [f.name for f in class_folders]
    
    def count_images(split_name):
        count = 0
        for class_folder in (split_dir / split_name).iterdir():
            if class_folder.is_dir():
                count += len(list(class_folder.glob('*.jpg'))) + \
                        len(list(class_folder.glob('*.jpeg'))) + \
                        len(list(class_folder.glob('*.png')))
        return count
    
    return {
        'num_classes': len(class_names),
        'class_names': class_names,
        'train_count': count_images('train'),
        'val_count': count_images('val'),
        'test_count': count_images('test'),
        'total_count': count_images('train') + count_images('val') + count_images('test')
    }


def get_preprocessing_fn(model_type='standard'):
    """
    Get preprocessing function for specific model type
    
    Args:
        model_type: 'standard', 'resnet50', 'vgg16', 'mobilenetv2', 'inceptionv3', 'densenet121'
        
    Returns:
        Preprocessing function
    """
    if model_type == 'standard':
        return lambda x: x / 255.0
    
    elif model_type == 'resnet50':
        from tensorflow.keras.applications.resnet50 import preprocess_input
        return preprocess_input
    
    elif model_type == 'vgg16':
        from tensorflow.keras.applications.vgg16 import preprocess_input
        return preprocess_input
    
    elif model_type == 'mobilenetv2':
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        return preprocess_input
    
    elif model_type == 'inceptionv3':
        from tensorflow.keras.applications.inception_v3 import preprocess_input
        return preprocess_input
    
    elif model_type == 'densenet121':
        from tensorflow.keras.applications.densenet import preprocess_input
        return preprocess_input
    
    else:
        return lambda x: x / 255.0


def create_tf_dataset(
    split_dir_path,
    split='train',
    img_size=(224, 224),
    batch_size=16,
    preprocessing='standard',
    shuffle=True,
    augment=False,
    cache=True,
    prefetch=True
):
    """
    Create optimized tf.data.Dataset from pre-split data
    
    Args:
        split_dir_path: Path to splitting_data folder
        split: 'train', 'val', or 'test'
        img_size: Target image size (height, width)
        batch_size: Batch size
        preprocessing: Preprocessing type ('standard', 'resnet50', etc.)
        shuffle: Shuffle dataset
        augment: Apply data augmentation (only for training)
        cache: Cache preprocessed data in memory
        prefetch: Prefetch batches for performance
        
    Returns:
        tf.data.Dataset, class_names
    """
    split_dir = Path(split_dir_path) / split
    
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    
    # Get class names
    class_folders = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    class_names = [f.name for f in class_folders]
    
    # Use Keras image_dataset_from_directory for efficiency
    dataset = tf.keras.utils.image_dataset_from_directory(
        split_dir,
        labels='inferred',
        label_mode='int',
        class_names=class_names,
        color_mode='rgb',
        batch_size=None,  # We'll batch later
        image_size=img_size,
        shuffle=shuffle,
        seed=42,
        interpolation='bilinear'
    )
    
    # Get preprocessing function
    preprocess_fn = get_preprocessing_fn(preprocessing)
    
    # Apply preprocessing
    def preprocess_image(image, label):
        image = tf.cast(image, tf.float32)
        image = preprocess_fn(image)
        return image, label
    
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Apply augmentation for training
    if augment and split == 'train':
        def augment_image(image, label):
            # Random flip
            image = tf.image.random_flip_left_right(image)
            # Random brightness
            image = tf.image.random_brightness(image, max_delta=0.1)
            # Random contrast
            image = tf.image.random_contrast (image, lower=0.9, upper=1.1)
            # Ensure values are in valid range
            image = tf.clip_by_value(image, -2.0, 2.0)  # For preprocessing that doesn't normalize to [0,1]
            return image, label
        
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Cache if requested (stores preprocessed data in RAM)
    if cache:
        dataset = dataset.cache()
    
    # Shuffle for training
    if shuffle and split == 'train':
        dataset = dataset.shuffle(buffer_size=1000, seed=42)
    
    # Batch
    dataset = dataset.batch(batch_size)
    
    # Prefetch for performance
    if prefetch:
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset, class_names


def create_datasets_for_training(
    split_dir_path,
    img_size=(224, 224),
    batch_size=16,
    preprocessing='standard',
    augment_train=True
):
    """
    Create train, validation, and test datasets
    
    Args:
        split_dir_path: Path to splitting_data folder  
        img_size: Target image size
        batch_size: Batch size
        preprocessing: Preprocessing type
        augment_train: Apply augmentation to training set
        
    Returns:
        train_ds, val_ds, test_ds, class_names
    """
    print("\n" + "="*80)
    print("📊 CREATING DATASETS FROM PRE-SPLIT DATA")
    print("="*80)
    print(f"   Split directory: {split_dir_path}")
    print(f"   Image size: {img_size}")
    print(f"   Batch size: {batch_size}")
    print(f"   Preprocessing: {preprocessing}")
    print(f"   Augmentation: {augment_train}")
    print()
    
    # Get dataset info
    info = get_dataset_info(split_dir_path)
    
    print(f"📂 Dataset Information:")
    print(f"   Classes: {info['num_classes']}")
    print(f"   Train samples: {info['train_count']:,}")
    print(f"   Val samples: {info['val_count']:,}")
    print(f"   Test samples: {info['test_count']:,}")
    print(f"   Total: {info['total_count']:,}")
    print()
    
    # Create datasets
    print("Creating training dataset...")
    train_ds, class_names = create_tf_dataset(
        split_dir_path,
        split='train',
        img_size=img_size,
        batch_size=batch_size,
        preprocessing=preprocessing,
        shuffle=True,
        augment=augment_train,
        cache=True,
        prefetch=True
    )
    
    print("Creating validation dataset...")
    val_ds, _ = create_tf_dataset(
        split_dir_path,
        split='val',
        img_size=img_size,
        batch_size=batch_size,
        preprocessing=preprocessing,
        shuffle=False,
        augment=False,
        cache=True,
        prefetch=True
    )
    
    print("Creating test dataset...")
    test_ds, _ = create_tf_dataset(
        split_dir_path,
        split='test',
        img_size=img_size,
        batch_size=batch_size,
        preprocessing=preprocessing,
        shuffle=False,
        augment=False,
        cache=True,
        prefetch=True
    )
    
    print("✅ All datasets created successfully!")
    print("="*80 + "\n")
    
    return train_ds, val_ds, test_ds, class_names, info


def extract_features_from_dataset(feature_extractor, dataset, total_samples, batch_size=16, desc="Extracting features"):
    """
    Extract features from a tf.data.Dataset using a feature extractor model
    Memory-efficient batch processing
    
    Args:
        feature_extractor: Keras model for feature extraction
        dataset: tf.data.Dataset
        total_samples: Total number of samples
        batch_size: Batch size
        desc: Description for progress
        
    Returns:
        features: numpy array of extracted features
        labels: numpy array of labels
    """
    print(f"\n{desc}...")
    print(f"   Total samples: {total_samples:,}")
    print(f"   Batch size: {batch_size}")
    
    features_list = []
    labels_list = []
    
    processed = 0
    
    for batch_images, batch_labels in dataset:
        # Extract features
        batch_features = feature_extractor.predict(batch_images, verbose=0)
        
        features_list.append(batch_features)
        labels_list.append(batch_labels.numpy())
        
        processed += len(batch_images)
        
        # Print progress
        if processed % (batch_size * 10) == 0 or processed >= total_samples:
            print(f"   Progress: {processed}/{total_samples} ({processed/total_samples*100:.1f}%)")
    
    # Concatenate all batches
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    
    print(f"✅ Feature extraction complete!")
    print(f"   Features shape: {features.shape}")
    print(f"   Labels shape: {labels.shape}\n")
    
    return features, labels


def test_data_pipeline(split_dir_path='dataset/praproses_result/splitting_data'):
    """
    Test the data pipeline to ensure it works correctly
    """
    print("\n" + "="*80)
    print("🧪 TESTING DATA PIPELINE")
    print("="*80 + "\n")
    
    # Test getting info
    print("1. Testing get_dataset_info()...")
    info = get_dataset_info(split_dir_path)
    print(f"   ✅ Found {info['num_classes']} classes")
    print(f"   ✅ Train: {info['train_count']:,} images")
    print(f"   ✅ Val: {info['val_count']:,} images")
    print(f"   ✅ Test: {info['test_count']:,} images\n")
    
    # Test creating dataset
    print("2. Testing create_tf_dataset()...")
    train_ds, class_names = create_tf_dataset(
        split_dir_path,
        split='train',
        batch_size=32,
        preprocessing='standard'
    )
    print(f"   ✅ Dataset created")
    print(f"   ✅ Class names: {len(class_names)} classes\n")
    
    # Test one batch
    print("3. Testing batch loading...")
    for images, labels in train_ds.take(1):
        print(f"   ✅ Batch shape: {images.shape}")
        print(f"   ✅ Labels shape: {labels.shape}")
        print(f"   ✅ Image dtype: {images.dtype}")
        print(f"   ✅ Image range: [{tf.reduce_min(images).numpy():.2f}, {tf.reduce_max(images).numpy():.2f}]")
    
    print("\n" + "="*80)
    print("✅ DATA PIPELINE TEST PASSED!")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Run test
    test_data_pipeline()
