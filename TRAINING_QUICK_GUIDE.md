# Quick Reference Guide - Training System

## 🚀 Quick Start Commands

### Start Training (Interactive Menu)
```bash
cd /home/fadhli/S2/facedet
./04_train_models.sh
```
OR
```bash
cd /home/fadhli/S2/facedet/training
python3 main_training.py
```

## 📋 Main Menu Options

When you run the main training system, you'll see:

```
1. Train all models          → Trains all 6 models sequentially
2. Train specific model      → Choose which model to train
3. View training history     → Browse all training reports
4. Compare all models        → Generate comparison charts
5. Test a trained model      → Evaluate on validation data
6. Analyze results          → Detailed statistical analysis
0. Exit
```

## 🤖 Available Models

| # | Model Name | Speed | Expected Accuracy | Best For |
|---|------------|-------|-------------------|----------|
| 1 | ResNet50 + SVM | Medium | 85-95% | High accuracy |
| 2 | VGG16 + SVM | Fast | 80-90% | Baseline |
| 3 | MobileNetV2 + SVM | **Fastest** | 80-92% | Mobile deployment |
| 4 | InceptionV3 + SVM | Slow | 85-95% | Multi-scale features |
| 5 | DenseNet121 + SVM | Medium | 85-95% | Parameter efficiency |
| 6 | CNN + Softmax | Slow | 75-88% | Baseline comparison |

**Recommendation**: Start with MobileNetV2 (fastest) to test the system.

## 📁 Important Directories

```
training/              → All training code
result_training/       → Saved models (.h5, .pkl)
report_training/       → Training reports (JSON, PNG)
```

## 🔧 Individual Model Training

```bash
cd /home/fadhli/S2/facedet

# Train specific models directly
python3 training/resnet50_svm.py
python3 training/vgg16_svm.py
python3 training/MobileNetV2_svm.py
python3 training/InceptionV3_svm.py
python3 training/DenseNet121_svm.py
python3 training/cnn_softmax.py
```

## 📊 View Results

### View Training Reports
```bash
cd training
python3 utils/view_reports.py

# Filter by model
python3 utils/view_reports.py --model ResNet50
```

### Compare All Models
```bash
cd training
python3 compare_models.py
```

### Test a Model
```bash
cd training
python3 test_models.py resnet50
python3 test_models.py mobilenetv2
python3 test_models.py cnn
```

## 📈 What Each Training Produces

For each model, you get:

### Saved Files
- **Feature Extractor**: `{model}_feature_extractor.h5`
- **SVM Classifier**: `{model}_svm_classifier.pkl`
- **Label Encoder**: `{model}_label_encoder.pkl`

### Report Files
- **JSON Report**: Complete metrics and configuration
- **Confusion Matrix**: PNG visualization
- **Training History**: PNG plot (CNN only)

### Report Contains
- ✅ Accuracy, Precision, Recall, F1-Score
- ✅ Per-class metrics for all 10 diseases
- ✅ Confusion matrix
- ✅ Training time
- ✅ Hyperparameters used

## ⚙️ Configuration

Edit `training/config.py` to customize:

```python
# Training parameters
TRAIN_SPLIT = 0.8           # 80% training, 20% validation
EPOCHS = 50                 # For CNN model
BATCH_SIZE = 32             # Reduce if out of memory

# SVM parameters
SVM_KERNEL = 'rbf'          # Kernel type
SVM_C = 1.0                 # Regularization
SVM_GAMMA = 'scale'         # Kernel coefficient
```

## 🔄 Workflow Example

### First Time Setup
```bash
# 1. Activate virtual environment
cd /home/fadhli/S2/facedet
source venv/bin/activate

# 2. Start training system
./04_train_models.sh

# 3. Choose option 2 (Train specific model)

# 4. Select model 3 (MobileNetV2 - fastest)

# 5. Wait for training to complete (~10-15 min)

# 6. View results
```

### Training All Models
```bash
# 1. Launch main menu
cd training
python3 main_training.py

# 2. Choose option 1 (Train all models)

# 3. Confirm (y)

# 4. Wait for completion (several hours)

# 5. Compare results (option 4 from menu)
```

### Viewing Historical Results
```bash
# If you've already trained models and want to see reports

# Method 1: From main menu
./04_train_models.sh
# Choose option 4

# Method 2: Direct command
cd training
python3 utils/view_reports.py
```

## 🎯 Common Tasks

### Task: "I want to train the best model"
**Answer**: Train ResNet50, DenseNet121, or InceptionV3
```bash
python3 training/resnet50_svm.py
```

### Task: "I want the fastest training"
**Answer**: Use MobileNetV2
```bash
python3 training/MobileNetV2_svm.py
```

### Task: "I want to compare all models"
**Answer**: First train all, then compare
```bash
cd training
python3 main_training.py  # Choose option 1, then option 4
```

### Task: "I want to see previous results"
**Answer**: View reports
```bash
cd training
python3 utils/view_reports.py
```

### Task: "I trained a model, now what?"
**Answer**: Test it, compare it, or use it
```bash
# Test the model
cd training
python3 test_models.py resnet50

# Compare with others
python3 compare_models.py

# Use in production (see deployment docs)
```

## 🐛 Troubleshooting Quick Fixes

### "Out of Memory"
```python
# Edit training/config.py
BATCH_SIZE = 16  # Reduce from 32 to 16
```

### "Model not found"
```bash
# Retrain the model
python3 training/resnet50_svm.py
```

### "Python not found"
```bash
# Use python3 instead of python
python3 training/main_training.py
```

### "Dataset not found"
```bash
# Run preprocessing first
python3 02_praproses.py
```

## 📚 Documentation Files

- `training/README.md` - Detailed documentation
- `walkthrough.md` - Complete implementation walkthrough
- This file - Quick reference

## ✅ Verification Checklist

Before training, verify:
- [ ] Virtual environment activated
- [ ] Dataset exists at `dataset/praproses_result/final_images/`
- [ ] At least 10 class folders in dataset
- [ ] GPU available (optional but recommended)

After training, verify:
- [ ] Models saved in `result_training/models/`
- [ ] Reports in `report_training/`
- [ ] Can view reports with `view_reports.py`
- [ ] Can compare models with `compare_models.py`

## 💡 Tips

1. **Start small**: Train MobileNetV2 first to verify everything works
2. **Use GPU**: Much faster training (automatic if available)
3. **Train overnight**: Run batch training before going to sleep
4. **Save reports**: All reports are automatically saved with timestamps
5. **Compare models**: Use comparison tool to find the best model
6. **Check history**: System remembers all training runs

## 🎓 Learning Resources

- See `training/README.md` for detailed explanations
- Check confusion matrices to understand model errors
- Use per-class metrics to identify problem classes
- Compare models to understand trade-offs

---

**Remember**: The interactive menu (`./04_train_models.sh`) is the easiest way to get started!
