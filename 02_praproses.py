# -*- coding: utf-8 -*-
"""
========================================================================================
SCRIPT PREPROCESSING IMAGE UNTUK KLASIFIKASI PENYAKIT KULIT
========================================================================================
Tahapan Preprocessing:
1. Denoising      - Menghilangkan gangguan visual
2. Resizing       - Menyamakan format input model
3. Enhancement    - Memperjelas fitur (CLAHE)
4. Augmentation   - Menambah variasi data (hanya training)
5. Normalization  - Menyetabilkan perhitungan matematis model
========================================================================================
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import shutil

# Konfigurasi Output untuk Jupyter Notebook
import warnings
warnings.filterwarnings('ignore')


# ========================================================================================
# KONFIGURASI GLOBAL
# ========================================================================================

class Config:
    """Konfigurasi untuk preprocessing"""
    
    # Path dataset
    INPUT_DIR = Path("dataset/original_dataset")
    OUTPUT_DIR = Path("dataset/praproses_result")
    
    # Parameter preprocessing
    TARGET_SIZE = (224, 224)  # Ukuran standar untuk model CNN
    
    # Denoising parameters
    DENOISE_H = 10  # Filter strength for luminance
    DENOISE_H_COLOR = 10  # Filter strength for color
    DENOISE_TEMPLATE_SIZE = 7  # Template patch size
    DENOISE_SEARCH_SIZE = 21  # Search window size
    
    # CLAHE parameters (Contrast Limited Adaptive Histogram Equalization)
    CLAHE_CLIP_LIMIT = 2.0
    CLAHE_TILE_SIZE = (8, 8)
    
    # Augmentation parameters (untuk training)
    AUGMENTATION_ENABLED = True
    AUGMENTATION_FACTOR = 2  # Berapa kali augmentasi per gambar
    
    # Normalization parameters
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
    NORMALIZE_STD = [0.229, 0.224, 0.225]   # ImageNet std


# ========================================================================================
# 📊 TAHAP 1: DENOISING - Menghilangkan Gangguan Visual
# ========================================================================================

def denoise_image(image):
    """
    Menghilangkan noise dari gambar menggunakan Non-Local Means Denoising
    
    Args:
        image: Image array (BGR format)
    
    Returns:
        Denoised image
    """
    # Gunakan fastNlMeansDenoisingColored untuk gambar berwarna
    denoised = cv2.fastNlMeansDenoisingColored(
        image,
        None,
        h=Config.DENOISE_H,
        hColor=Config.DENOISE_H_COLOR,
        templateWindowSize=Config.DENOISE_TEMPLATE_SIZE,
        searchWindowSize=Config.DENOISE_SEARCH_SIZE
    )
    return denoised


def process_denoising(input_dir, output_dir):
    """
    Proses denoising untuk semua gambar dalam dataset
    
    Returns:
        Dictionary berisi statistik proses
    """
    print("\n" + "="*80)
    print("📍 TAHAP 1/5: DENOISING - Menghilangkan Gangguan Visual")
    print("="*80)
    
    stats = {
        'total_images': 0,
        'processed': 0,
        'failed': 0,
        'classes': []
    }
    
    # Buat direktori output
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dapatkan semua kelas
    class_folders = sorted([f for f in input_dir.iterdir() if f.is_dir()])
    
    # Hitung total gambar untuk progress bar
    total_images = 0
    for class_folder in class_folders:
        image_files = list(class_folder.glob('*.jpg')) + list(class_folder.glob('*.jpeg')) + \
                      list(class_folder.glob('*.png')) + list(class_folder.glob('*.bmp'))
        total_images += len(image_files)
    
    stats['total_images'] = total_images
    
    # Progress bar untuk seluruh proses
    pbar = tqdm(total=total_images, desc="🔄 Denoising Progress", 
                unit="img", colour="blue")
    
    for class_folder in class_folders:
        class_name = class_folder.name
        stats['classes'].append(class_name)
        
        # Buat folder output untuk kelas ini
        output_class_dir = output_dir / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Proses setiap gambar
        image_files = list(class_folder.glob('*.jpg')) + list(class_folder.glob('*.jpeg')) + \
                      list(class_folder.glob('*.png')) + list(class_folder.glob('*.bmp'))
        
        for img_path in image_files:
            try:
                # Baca gambar
                image = cv2.imread(str(img_path))
                
                if image is None:
                    stats['failed'] += 1
                    pbar.update(1)
                    continue
                
                # Terapkan denoising
                denoised = denoise_image(image)
                
                # Simpan hasil
                output_path = output_class_dir / img_path.name
                cv2.imwrite(str(output_path), denoised)
                
                stats['processed'] += 1
                
            except Exception as e:
                stats['failed'] += 1
                print(f"\n⚠️ Error processing {img_path.name}: {str(e)}")
            
            pbar.update(1)
    
    pbar.close()
    
    # Tampilkan ringkasan
    print(f"\n✅ Denoising selesai!")
    print(f"   📊 Total: {stats['total_images']} | Berhasil: {stats['processed']} | Gagal: {stats['failed']}")
    print(f"   💾 Output: {output_dir}")
    
    return stats


# ========================================================================================
# 📏 TAHAP 2: RESIZING - Menyamakan Format Input Model
# ========================================================================================

def resize_image(image, target_size=Config.TARGET_SIZE):
    """
    Resize gambar ke ukuran target dengan mempertahankan aspect ratio
    
    Args:
        image: Image array
        target_size: Tuple (width, height)
    
    Returns:
        Resized image
    """
    # Resize dengan interpolasi INTER_AREA untuk downscaling (lebih baik)
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    return resized


def process_resizing(input_dir, output_dir):
    """
    Proses resizing untuk semua gambar
    """
    print("\n" + "="*80)
    print(f"📍 TAHAP 2/5: RESIZING - Menyamakan Format ke {Config.TARGET_SIZE}")
    print("="*80)
    
    stats = {
        'total_images': 0,
        'processed': 0,
        'failed': 0
    }
    
    # Buat direktori output
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dapatkan semua kelas
    class_folders = sorted([f for f in input_dir.iterdir() if f.is_dir()])
    
    # Hitung total gambar
    total_images = 0
    for class_folder in class_folders:
        image_files = list(class_folder.glob('*.jpg')) + list(class_folder.glob('*.jpeg')) + \
                      list(class_folder.glob('*.png')) + list(class_folder.glob('*.bmp'))
        total_images += len(image_files)
    
    stats['total_images'] = total_images
    
    # Progress bar
    pbar = tqdm(total=total_images, desc="📐 Resizing Progress", 
                unit="img", colour="green")
    
    for class_folder in class_folders:
        class_name = class_folder.name
        
        # Buat folder output
        output_class_dir = output_dir / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Proses gambar
        image_files = list(class_folder.glob('*.jpg')) + list(class_folder.glob('*.jpeg')) + \
                      list(class_folder.glob('*.png')) + list(class_folder.glob('*.bmp'))
        
        for img_path in image_files:
            try:
                # Baca gambar
                image = cv2.imread(str(img_path))
                
                if image is None:
                    stats['failed'] += 1
                    pbar.update(1)
                    continue
                
                # Resize
                resized = resize_image(image)
                
                # Simpan
                output_path = output_class_dir / img_path.name
                cv2.imwrite(str(output_path), resized)
                
                stats['processed'] += 1
                
            except Exception as e:
                stats['failed'] += 1
                print(f"\n⚠️ Error resizing {img_path.name}: {str(e)}")
            
            pbar.update(1)
    
    pbar.close()
    
    print(f"\n✅ Resizing selesai!")
    print(f"   📊 Total: {stats['total_images']} | Berhasil: {stats['processed']} | Gagal: {stats['failed']}")
    print(f"   💾 Output: {output_dir}")
    
    return stats


# ========================================================================================
# ✨ TAHAP 3: ENHANCEMENT - Memperjelas Fitur dengan CLAHE
# ========================================================================================

def enhance_image_clahe(image):
    """
    Enhance gambar menggunakan CLAHE (Contrast Limited Adaptive Histogram Equalization)
    
    Args:
        image: Image array (BGR format)
    
    Returns:
        Enhanced image
    """
    # Konversi ke LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE pada L channel
    clahe = cv2.createCLAHE(
        clipLimit=Config.CLAHE_CLIP_LIMIT,
        tileGridSize=Config.CLAHE_TILE_SIZE
    )
    l_clahe = clahe.apply(l)
    
    # Merge kembali
    lab_clahe = cv2.merge([l_clahe, a, b])
    
    # Konversi kembali ke BGR
    enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    return enhanced


def process_enhancement(input_dir, output_dir):
    """
    Proses enhancement untuk semua gambar
    """
    print("\n" + "="*80)
    print("📍 TAHAP 3/5: ENHANCEMENT - Memperjelas Fitur dengan CLAHE")
    print("="*80)
    
    stats = {
        'total_images': 0,
        'processed': 0,
        'failed': 0
    }
    
    # Buat direktori output
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dapatkan semua kelas
    class_folders = sorted([f for f in input_dir.iterdir() if f.is_dir()])
    
    # Hitung total gambar
    total_images = 0
    for class_folder in class_folders:
        image_files = list(class_folder.glob('*.jpg')) + list(class_folder.glob('*.jpeg')) + \
                      list(class_folder.glob('*.png')) + list(class_folder.glob('*.bmp'))
        total_images += len(image_files)
    
    stats['total_images'] = total_images
    
    # Progress bar
    pbar = tqdm(total=total_images, desc="✨ Enhancement Progress", 
                unit="img", colour="yellow")
    
    for class_folder in class_folders:
        class_name = class_folder.name
        
        # Buat folder output
        output_class_dir = output_dir / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Proses gambar
        image_files = list(class_folder.glob('*.jpg')) + list(class_folder.glob('*.jpeg')) + \
                      list(class_folder.glob('*.png')) + list(class_folder.glob('*.bmp'))
        
        for img_path in image_files:
            try:
                # Baca gambar
                image = cv2.imread(str(img_path))
                
                if image is None:
                    stats['failed'] += 1
                    pbar.update(1)
                    continue
                
                # Enhancement
                enhanced = enhance_image_clahe(image)
                
                # Simpan
                output_path = output_class_dir / img_path.name
                cv2.imwrite(str(output_path), enhanced)
                
                stats['processed'] += 1
                
            except Exception as e:
                stats['failed'] += 1
                print(f"\n⚠️ Error enhancing {img_path.name}: {str(e)}")
            
            pbar.update(1)
    
    pbar.close()
    
    print(f"\n✅ Enhancement selesai!")
    print(f"   📊 Total: {stats['total_images']} | Berhasil: {stats['processed']} | Gagal: {stats['failed']}")
    print(f"   💾 Output: {output_dir}")
    
    return stats


# ========================================================================================
# 🔄 TAHAP 4: AUGMENTATION - Menambah Variasi Data (Training Only)
# ========================================================================================

def augment_image(image, augment_type='flip'):
    """
    Augmentasi gambar dengan berbagai transformasi
    
    Args:
        image: Image array
        augment_type: Tipe augmentasi ('flip', 'rotate', 'brightness', 'zoom')
    
    Returns:
        Augmented image
    """
    if augment_type == 'flip_horizontal':
        return cv2.flip(image, 1)
    
    elif augment_type == 'flip_vertical':
        return cv2.flip(image, 0)
    
    elif augment_type == 'rotate_90':
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    
    elif augment_type == 'rotate_180':
        return cv2.rotate(image, cv2.ROTATE_180)
    
    elif augment_type == 'rotate_270':
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    elif augment_type == 'brightness_increase':
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.add(hsv[:, :, 2], 30)  # Increase brightness
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    elif augment_type == 'brightness_decrease':
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.subtract(hsv[:, :, 2], 30)  # Decrease brightness
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    else:
        return image


def process_augmentation(input_dir, output_dir):
    """
    Proses augmentasi untuk menambah variasi data
    """
    print("\n" + "="*80)
    print("📍 TAHAP 4/5: AUGMENTATION - Menambah Variasi Data")
    print("="*80)
    
    if not Config.AUGMENTATION_ENABLED:
        print("⏭️  Augmentasi dinonaktifkan (set AUGMENTATION_ENABLED=True untuk mengaktifkan)")
        return {'skipped': True}
    
    stats = {
        'total_images': 0,
        'augmented_images': 0,
        'processed': 0,
        'failed': 0
    }
    
    # Buat direktori output
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Tipe augmentasi yang akan diterapkan
    augmentation_types = [
        'flip_horizontal',
        'rotate_90',
        'brightness_increase',
        'brightness_decrease'
    ]
    
    # Dapatkan semua kelas
    class_folders = sorted([f for f in input_dir.iterdir() if f.is_dir()])
    
    # Hitung total gambar original
    total_images = 0
    for class_folder in class_folders:
        image_files = list(class_folder.glob('*.jpg')) + list(class_folder.glob('*.jpeg')) + \
                      list(class_folder.glob('*.png')) + list(class_folder.glob('*.bmp'))
        total_images += len(image_files)
    
    stats['total_images'] = total_images
    
    # Total operasi = original + augmented
    total_operations = total_images * (1 + len(augmentation_types))
    
    # Progress bar
    pbar = tqdm(total=total_operations, desc="🔄 Augmentation Progress", 
                unit="img", colour="magenta")
    
    for class_folder in class_folders:
        class_name = class_folder.name
        
        # Buat folder output
        output_class_dir = output_dir / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Proses gambar
        image_files = list(class_folder.glob('*.jpg')) + list(class_folder.glob('*.jpeg')) + \
                      list(class_folder.glob('*.png')) + list(class_folder.glob('*.bmp'))
        
        for img_path in image_files:
            try:
                # Baca gambar original
                image = cv2.imread(str(img_path))
                
                if image is None:
                    stats['failed'] += 1
                    pbar.update(1 + len(augmentation_types))
                    continue
                
                # Simpan original
                output_path = output_class_dir / img_path.name
                cv2.imwrite(str(output_path), image)
                stats['processed'] += 1
                pbar.update(1)
                
                # Buat augmentasi
                for i, aug_type in enumerate(augmentation_types):
                    augmented = augment_image(image, aug_type)
                    
                    # Buat nama file baru
                    stem = img_path.stem
                    suffix = img_path.suffix
                    aug_filename = f"{stem}_aug_{aug_type}{suffix}"
                    
                    # Simpan augmented
                    aug_output_path = output_class_dir / aug_filename
                    cv2.imwrite(str(aug_output_path), augmented)
                    
                    stats['augmented_images'] += 1
                    pbar.update(1)
                
            except Exception as e:
                stats['failed'] += 1
                print(f"\n⚠️ Error augmenting {img_path.name}: {str(e)}")
                pbar.update(1 + len(augmentation_types))
    
    pbar.close()
    
    print(f"\n✅ Augmentation selesai!")
    print(f"   📊 Original: {stats['processed']} | Augmented: {stats['augmented_images']} | Total: {stats['processed'] + stats['augmented_images']}")
    print(f"   💾 Output: {output_dir}")
    
    return stats


# ========================================================================================
# 📊 TAHAP 5: NORMALIZATION - Menyetabilkan Perhitungan Matematis
# ========================================================================================

def normalize_image(image):
    """
    Normalisasi gambar menggunakan ImageNet mean dan std
    
    Args:
        image: Image array (BGR format)
    
    Returns:
        Normalized image (float32, range -1 to 1)
    """
    # Konversi ke RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Konversi ke float dan scale ke [0, 1]
    image_float = image_rgb.astype(np.float32) / 255.0
    
    # Normalisasi dengan ImageNet mean dan std
    mean = np.array(Config.NORMALIZE_MEAN, dtype=np.float32)
    std = np.array(Config.NORMALIZE_STD, dtype=np.float32)
    
    normalized = (image_float - mean) / std
    
    return normalized


def save_normalized_data(input_dir, output_file):
    """
    Simpan data yang sudah dinormalisasi ke dalam format .npz
    
    Args:
        input_dir: Direktori input dengan gambar yang sudah dipreprocess
        output_file: Path file output .npz
    """
    print("\n" + "="*80)
    print("📍 TAHAP 5/5: NORMALIZATION - Menyetabilkan Perhitungan Matematis")
    print("="*80)
    
    # Data containers
    X_data = []
    y_data = []
    class_names = []
    
    # Dapatkan semua kelas
    class_folders = sorted([f for f in input_dir.iterdir() if f.is_dir()])
    
    # Mapping kelas ke index
    class_to_idx = {folder.name: idx for idx, folder in enumerate(class_folders)}
    class_names = [folder.name for folder in class_folders]
    
    # Hitung total gambar
    total_images = 0
    for class_folder in class_folders:
        image_files = list(class_folder.glob('*.jpg')) + list(class_folder.glob('*.jpeg')) + \
                      list(class_folder.glob('*.png')) + list(class_folder.glob('*.bmp'))
        total_images += len(image_files)
    
    # Progress bar
    pbar = tqdm(total=total_images, desc="📊 Normalization Progress", 
                unit="img", colour="cyan")
    
    processed = 0
    failed = 0
    
    for class_folder in class_folders:
        class_name = class_folder.name
        class_idx = class_to_idx[class_name]
        
        # Proses gambar
        image_files = list(class_folder.glob('*.jpg')) + list(class_folder.glob('*.jpeg')) + \
                      list(class_folder.glob('*.png')) + list(class_folder.glob('*.bmp'))
        
        for img_path in image_files:
            try:
                # Baca gambar
                image = cv2.imread(str(img_path))
                
                if image is None:
                    failed += 1
                    pbar.update(1)
                    continue
                
                # Normalisasi
                normalized = normalize_image(image)
                
                # Simpan ke list
                X_data.append(normalized)
                y_data.append(class_idx)
                
                processed += 1
                
            except Exception as e:
                failed += 1
                print(f"\n⚠️ Error normalizing {img_path.name}: {str(e)}")
            
            pbar.update(1)
    
    pbar.close()
    
    # Konversi ke numpy arrays
    X_data = np.array(X_data, dtype=np.float32)
    y_data = np.array(y_data, dtype=np.int32)
    
    # Simpan ke file .npz
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        output_path,
        X=X_data,
        y=y_data,
        class_names=class_names,
        class_to_idx=class_to_idx
    )
    
    print(f"\n✅ Normalization selesai!")
    print(f"   📊 Total: {total_images} | Berhasil: {processed} | Gagal: {failed}")
    print(f"   📦 Data shape: X={X_data.shape}, y={y_data.shape}")
    print(f"   💾 Output: {output_path}")
    
    return {
        'total_images': total_images,
        'processed': processed,
        'failed': failed,
        'X_shape': X_data.shape,
        'y_shape': y_data.shape,
        'num_classes': len(class_names),
        'class_names': class_names
    }


# ========================================================================================
# 🚀 FUNGSI UTAMA - Menjalankan Semua Tahap Preprocessing
# ========================================================================================

def run_full_preprocessing(visualize_samples=True):
    """
    Menjalankan semua tahap preprocessing secara berurutan
    
    Args:
        visualize_samples: Apakah akan menampilkan sample hasil preprocessing
    
    Returns:
        Dictionary berisi semua statistik
    """
    print("\n" + "="*80)
    print("🚀 MEMULAI PREPROCESSING PIPELINE")
    print("="*80)
    print(f"📂 Input: {Config.INPUT_DIR}")
    print(f"📂 Output: {Config.OUTPUT_DIR}")
    print(f"🎯 Target Size: {Config.TARGET_SIZE}")
    print(f"🔄 Augmentation: {'Enabled' if Config.AUGMENTATION_ENABLED else 'Disabled'}")
    print("="*80)
    
    all_stats = {}
    
    # Direktori sementara untuk setiap tahap
    temp_dirs = {
        'denoised': Config.OUTPUT_DIR / 'temp_1_denoised',
        'resized': Config.OUTPUT_DIR / 'temp_2_resized',
        'enhanced': Config.OUTPUT_DIR / 'temp_3_enhanced',
        'augmented': Config.OUTPUT_DIR / 'temp_4_augmented'
    }
    
    try:
        # TAHAP 1: Denoising
        all_stats['denoising'] = process_denoising(
            Config.INPUT_DIR,
            temp_dirs['denoised']
        )
        
        # TAHAP 2: Resizing
        all_stats['resizing'] = process_resizing(
            temp_dirs['denoised'],
            temp_dirs['resized']
        )
        
        # TAHAP 3: Enhancement
        all_stats['enhancement'] = process_enhancement(
            temp_dirs['resized'],
            temp_dirs['enhanced']
        )
        
        # TAHAP 4: Augmentation
        all_stats['augmentation'] = process_augmentation(
            temp_dirs['enhanced'],
            temp_dirs['augmented']
        )
        
        # TAHAP 5: Normalization
        all_stats['normalization'] = save_normalized_data(
            temp_dirs['augmented'],
            Config.OUTPUT_DIR / 'preprocessed_data.npz'
        )
        
        # Pindahkan hasil akhir
        final_dir = Config.OUTPUT_DIR / 'final_images'
        if final_dir.exists():
            shutil.rmtree(final_dir)
        shutil.copytree(temp_dirs['augmented'], final_dir)
        
        print("\n" + "="*80)
        print("✅ PREPROCESSING SELESAI!")
        print("="*80)
        
        # Cleanup temporary directories
        print("\n🧹 Membersihkan file sementara...")
        for temp_dir in temp_dirs.values():
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        print("✅ Cleanup selesai!")
        
        # Summary
        print("\n" + "="*80)
        print("📊 RINGKASAN PREPROCESSING")
        print("="*80)
        
        if 'normalization' in all_stats:
            norm_stats = all_stats['normalization']
            print(f"Total Dataset: {norm_stats['processed']:,} gambar")
            print(f"Jumlah Kelas: {norm_stats['num_classes']}")
            print(f"Shape Data: X={norm_stats['X_shape']}, y={norm_stats['y_shape']}")
            print(f"\nDaftar Kelas:")
            for i, class_name in enumerate(norm_stats['class_names']):
                print(f"  {i}. {class_name}")
        
        print("\n💾 Output Files:")
        print(f"  - Final Images: {final_dir}")
        print(f"  - Normalized Data: {Config.OUTPUT_DIR / 'preprocessed_data.npz'}")
        print("="*80)
        
        # Visualisasi sample (opsional)
        if visualize_samples:
            visualize_preprocessing_samples(final_dir)
        
    except Exception as e:
        print(f"\n❌ Error during preprocessing: {str(e)}")
        raise
    
    return all_stats


# ========================================================================================
# 📸 VISUALISASI SAMPLE HASIL PREPROCESSING
# ========================================================================================

def visualize_preprocessing_samples(processed_dir, num_samples=5):
    """
    Menampilkan sample gambar hasil preprocessing
    """
    print("\n" + "="*80)
    print("📸 VISUALISASI SAMPLE HASIL PREPROCESSING")
    print("="*80)
    
    # Ambil satu kelas secara random
    class_folders = [f for f in processed_dir.iterdir() if f.is_dir()]
    if not class_folders:
        print("⚠️ Tidak ada folder kelas ditemukan")
        return
    
    sample_class = class_folders[0]
    print(f"Menampilkan sample dari kelas: {sample_class.name}")
    
    # Ambil beberapa gambar
    image_files = list(sample_class.glob('*.jpg')) + list(sample_class.glob('*.png'))
    sample_files = image_files[:num_samples]
    
    if not sample_files:
        print("⚠️ Tidak ada gambar ditemukan")
        return
    
    # Buat display
    fig, axes = plt.subplots(1, len(sample_files), figsize=(15, 3))
    
    if len(sample_files) == 1:
        axes = [axes]
    
    for ax, img_path in zip(axes, sample_files):
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        ax.imshow(img_rgb)
        ax.set_title(img_path.stem[:20], fontsize=8)
        ax.axis('off')
    
    plt.tight_layout()
    
    # Simpan visualisasi
    output_path = Config.OUTPUT_DIR / 'sample_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualisasi disimpan di: {output_path}")
    
    plt.show()


# ========================================================================================
# 🎯 MAIN EXECUTION
# ========================================================================================

if __name__ == "__main__":
    # Jalankan preprocessing lengkap
    stats = run_full_preprocessing(visualize_samples=True)
