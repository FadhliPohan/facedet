# Training System Documentation

## Overview
Comprehensive training system for skin disease detection using multiple deep learning architectures with automated reporting, history tracking, and model comparison.

## Supported Models

### 1. ResNet50 + SVM
- **Architecture**: ResNet50 (ImageNet pretrained) + SVM classifier
- **Input Size**: 224×224
- **Features**: Deep residual learning with skip connections
- **Use Case**: Best for complex feature extraction

### 2. VGG16 + SVM
- **Architecture**: VGG16 (ImageNet pretrained) + SVM classifier
- **Input Size**: 224×224
- **Features**: Simple and proven architecture
- **Use Case**: Good baseline model

### 3. MobileNetV2 + SVM
- **Architecture**: MobileNetV2 (ImageNet pretrained) + SVM classifier
- **Input Size**: 224×224
- **Features**: Lightweight and efficient
- **Use Case**: Best for deployment on mobile/edge devices

### 4. InceptionV3 + SVM
- **Architecture**: InceptionV3 (ImageNet pretrained) + SVM classifier
- **Input Size**: 299×299
- **Features**: Multi-scale feature extraction
- **Use Case**: Good for complex patterns

### 5. DenseNet121 + SVM
- **Architecture**: DenseNet121 (ImageNet pretrained) + SVM classifier
- **Input Size**: 224×224
- **Features**: Dense connections for feature reuse
- **Use Case**: Efficient parameter usage

### 6. Custom CNN + Softmax
- **Architecture**: Custom 4-layer CNN trained from scratch
- **Input Size**: 224×224
- **Features**: Batch normalization, dropout, data augmentation
- **Use Case**: Baseline comparison

## Directory Structure

```
training/
├── config.py                    # Configuration settings
├── main_training.py             # Main interactive menu
├── resnet50_svm.py             # ResNet50 training script
├── vgg16_svm.py                # VGG16 training script
├── MobileNetV2_svm.py          # MobileNetV2 training script
├── InceptionV3_svm.py          # InceptionV3 training script
├── DenseNet121_svm.py          # DenseNet121 training script
├── cnn_softmax.py              # CNN training script
├── compare_models.py           # Model comparison tool
├── test_models.py              # Model testing tool
└── utils/
    ├── __init__.py
    ├── data_loader.py          # Data loading utilities
    ├── model_builder.py        # Model building utilities
    ├── metrics_calculator.py   # Metrics calculation
    ├── report_generator.py     # Report generation
    └── view_reports.py         # Report viewing

result_training/
└── models/                      # Trained models (.h5, .pkl)

report_training/                 # Training reports (JSON, PNG)
```

## Quick Start

### 1. Interactive Menu (Recommended)

```bash
cd training
python main_training.py
```

This will show an interactive menu with options to:
1. Train all models
2. Train specific model
3. View training history/reports
4. Compare all models
5. Test a trained model
6. Analyze results

### 2. Train Individual Models

```bash
# Train ResNet50
python training/resnet50_svm.py

# Train VGG16
python training/vgg16_svm.py

# Train MobileNetV2
python training/MobileNetV2_svm.py

# Train InceptionV3
python training/InceptionV3_svm.py

# Train DenseNet121
python training/DenseNet121_svm.py

# Train CNN
python training/cnn_softmax.py
```

### 3. Compare Models

```bash
python training/compare_models.py
```

### 4. View Reports

```bash
python training/utils/view_reports.py

# Filter by model
python training/utils/view_reports.py --model ResNet50

# Hide per-class metrics
python training/utils/view_reports.py --no-per-class
```

### 5. Test Models

```bash
python training/test_models.py resnet50
python training/test_models.py vgg16
python training/test_models.py mobilenetv2
python training/test_models.py inceptionv3
python training/test_models.py densenet121
python training/test_models.py cnn
```

## Configuration

Edit `training/config.py` to customize:

### Paths
- `DATASET_DIR`: Path to preprocessed images
- `MODEL_DIR`: Where to save trained models
- `REPORT_DIR`: Where to save training reports

### Training Parameters
```python
TRAIN_SPLIT = 0.8           # Train/validation split
VAL_SPLIT = 0.2
RANDOM_SEED = 42            # For reproducibility
EPOCHS = 50                 # For CNN model
EARLY_STOPPING_PATIENCE = 10
```

### SVM Parameters
```python
SVM_KERNEL = 'rbf'          # SVM kernel type
SVM_C = 1.0                 # Regularization parameter
SVM_GAMMA = 'scale'         # Kernel coefficient
```

## Output Files

### Models
Each training run saves:
- **Feature Extractor** (`.h5`): Pretrained CNN for feature extraction
- **SVM Classifier** (`.pkl`): Trained SVM model
- **Label Encoder** (`.pkl`): Label encoding mapping
- **CNN Model** (`.h5`): Complete CNN model (for CNN-Softmax)

Example:
```
result_training/models/
├── resnet50_feature_extractor.h5
├── resnet50_svm_classifier.pkl
├── resnet50_label_encoder.pkl
├── vgg16_feature_extractor.h5
├── vgg16_svm_classifier.pkl
└── ...
```

### Reports
Each training run generates:
- **JSON Report**: Complete metrics and configuration
- **Confusion Matrix** (PNG): Visual representation
- **Training History** (PNG): For CNN model

Example JSON Report:
```json
{
  "model_info": {
    "model_name": "ResNet50_SVM",
    "model_type": "ResNet50 + SVM",
    "timestamp": "2026-02-06_17-30-45",
    "training_time_seconds": 1234.56,
    "training_time_formatted": "20m 34s"
  },
  "dataset_info": {
    "num_classes": 10,
    "class_names": ["Eczema", "Melanoma", ...],
    "num_samples_train": 32000,
    "num_samples_val": 8000
  },
  "metrics": {
    "accuracy": 95.67,
    "precision": 95.23,
    "recall": 95.45,
    "f1_score": 95.34
  },
  "confusion_matrix": [[...], ...],
  "per_class_metrics": {...},
  "hyperparameters": {...}
}
```

## Features

### 1. Resume Training Detection
- Automatically detects existing training reports
- Asks before retraining
- Shows previous results

### 2. Comprehensive Metrics
- **Overall**: Accuracy, Precision, Recall, F1-Score
- **Per-Class**: Individual metrics for each disease
- **Confusion Matrix**: Detailed classification breakdown

### 3. Model Comparison
- Side-by-side metric comparison
- Performance ranking
- Visualizations:
  - Metrics bar charts
  - Radar chart
  - Training time comparison

### 4. History Tracking
- All training runs saved to JSON
- Searchable and filterable
- Can view historical performance

### 5. Testing Framework
- Test trained models on validation data
- Generate test reports
- Compare test vs. training performance

## Best Practices

### 1. Training Order
Recommended order from fastest to slowest:
1. MobileNetV2 (fastest)
2. VGG16
3. ResNet50
4. DenseNet121
5. InceptionV3
6. CNN (depends on epochs)

### 2. Hardware Requirements
- **Minimum**: 8GB RAM, CPU training
- **Recommended**: 16GB+ RAM, NVIDIA GPU (8GB+ VRAM)
- **Optimal**: 32GB RAM, NVIDIA GPU (16GB+ VRAM)

### 3. GPU Memory Management
The scripts automatically enable GPU memory growth to prevent OOM errors.

### 4. Batch Training
When training all models:
- Use `main_training.py` → Option 1
- Run overnight on good hardware
- Monitor first model to ensure no errors

## Troubleshooting

### Out of Memory (OOM)
```python
# Reduce batch size in config.py
BATCH_SIZE = 16  # or 8

# Or load subset for testing
X, y, class_names, label_encoder = load_dataset(max_samples_per_class=100)
```

### Model Not Found
```python
# Check model files exist:
ls result_training/models/

# Retrain if missing:
python training/resnet50_svm.py
```

### Report Viewing Issues
```python
# Check reports exist:
ls report_training/

# Generate new report by retraining
```

## Performance Expectations

Based on validation set (20% of data):

| Model | Expected Accuracy | Training Time* |
|-------|------------------|----------------|
| ResNet50 + SVM | 85-95% | 15-30 min |
| VGG16 + SVM | 80-90% | 10-20 min |
| MobileNetV2 + SVM | 80-92% | 8-15 min |
| InceptionV3 + SVM | 85-95% | 20-35 min |
| DenseNet121 + SVM | 85-95% | 15-30 min |
| CNN + Softmax | 75-88% | 30-60 min |

*Times are approximate for ~40,000 images on NVIDIA GPU

## Advanced Usage

### Custom Test Data
Modify `test_models.py` to load your own test set:

```python
def load_custom_test_data(test_dir):
    # Implement custom loading logic
    pass
```

### Hyperparameter Tuning
Edit `config.py` or create custom training scripts:

```python
# Example: Grid search for SVM
for C in [0.1, 1.0, 10.0]:
    for gamma in ['scale', 'auto', 0.01]:
        svm = SVC(kernel='rbf', C=C, gamma=gamma)
        # Train and evaluate
```

### Export Results
All reports are in JSON format and can be easily parsed:

```python
import json
import pandas as pd

# Load all reports
reports = []
for report_file in Path('report_training').glob('*.json'):
    with open(report_file) as f:
        reports.append(json.load(f))

# Convert to DataFrame
df = pd.DataFrame([r['metrics'] for r in reports])
df.to_csv('all_results.csv')
```

## Support

For issues or questions:
1. Check this documentation
2. Review error messages in console
3. Check log files (if implemented)
4. Review configuration in `config.py`
