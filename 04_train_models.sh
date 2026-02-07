#!/bin/bash
# Quick start script for training system

echo "================================================================================================"
echo "🚀 SKIN DISEASE DETECTION - TRAINING SYSTEM QUICK START"
echo "================================================================================================"
echo ""
echo "This script will help you get started with the training system."
echo ""

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Warning: Virtual environment not activated!"
    echo "   It's recommended to activate your virtual environment first:"
    echo "   source venv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/n): " continue
    if [[ "$continue" != "y" ]]; then
        echo "Exiting..."
        exit 0
    fi
fi

# Check if dataset exists
if [ ! -d "dataset/praproses_result/final_images" ]; then
    echo "❌ Dataset not found: dataset/praproses_result/final_images"
    echo "   Please run preprocessing first (02_praproses.py)"
    exit 1
fi

echo "✅ Dataset found!"
echo ""

# Display dataset info
cd dataset/praproses_result/final_images
num_classes=$(ls -d */ | wc -l)
total_images=$(find . -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) | wc -l)
cd ../../..

echo "📊 Dataset Information:"
echo "   Classes: $num_classes"
echo "   Total Images: $total_images"
echo ""

# Show options
echo "================================================================================================"
echo "📋 SELECT AN OPTION:"
echo "================================================================================================"
echo ""
echo "  1. Launch Interactive Training Menu (Recommended)"
echo "  2. Train a single model (Choose from list)"
echo "  3. Train all models (Takes several hours)"
echo "  4. View existing training reports"
echo "  5. Compare trained models"
echo "  0. Exit"
echo ""
read -p "Enter your choice (0-5): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Launching interactive training menu..."
        echo ""
        cd training
        python main_training.py
        ;;
    2)
        echo ""
        echo "📋 SELECT MODEL TO TRAIN:"
        echo "  1. ResNet50 + SVM"
        echo "  2. VGG16 + SVM"
        echo "  3. MobileNetV2 + SVM"
        echo "  4. InceptionV3 + SVM"
        echo "  5. DenseNet121 + SVM"
        echo "  6. CNN + Softmax"
        echo ""
        read -p "Enter model number (1-6): " model_choice
        
        case $model_choice in
            1) python training/resnet50_svm.py ;;
            2) python training/vgg16_svm.py ;;
            3) python training/MobileNetV2_svm.py ;;
            4) python training/InceptionV3_svm.py ;;
            5) python training/DenseNet121_svm.py ;;
            6) python training/cnn_softmax.py ;;
            *) echo "Invalid choice!"; exit 1 ;;
        esac
        ;;
    3)
        echo ""
        echo "⚠️  WARNING: This will train all 6 models sequentially."
        echo "   This may take several hours depending on your hardware."
        echo ""
        read -p "Are you sure you want to continue? (y/n): " confirm
        if [[ "$confirm" == "y" ]]; then
            cd training
            python main_training.py
            # Will trigger option 1 in menu
        else
            echo "Cancelled."
        fi
        ;;
    4)
        echo ""
        echo "📋 Viewing training reports..."
        echo ""
        cd training
        python utils/view_reports.py
        ;;
    5)
        echo ""
        echo "📊 Comparing trained models..."
        echo ""
        cd training
        python compare_models.py
        ;;
    0)
        echo ""
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo ""
        echo "❌ Invalid choice!"
        exit 1
        ;;
esac

echo ""
echo "================================================================================================"
echo "✅ Done!"
echo "================================================================================================"
echo ""
echo "📁 Output locations:"
echo "   Models: result_training/models/"
echo "   Reports: report_training/"
echo ""
echo "For more information, see: training/README.md"
echo ""
