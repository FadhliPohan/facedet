# 📚 PANDUAN PENGGUNAAN PREPROCESSING DI JUPYTER NOTEBOOK

## 🚀 Cara Menggunakan Script di Jupyter Notebook

### 1️⃣ Import Module dan Konfigurasi

```python
# Import semua fungsi dari script preprocessing
from 02_praproses import *

# OPSIONAL: Ubah konfigurasi jika diperlukan
Config.TARGET_SIZE = (224, 224)  # Ukuran resize
Config.AUGMENTATION_ENABLED = True  # Aktifkan/nonaktifkan augmentasi
Config.AUGMENTATION_FACTOR = 2  # Berapa kali augmentasi
```

### 2️⃣ Jalankan Preprocessing Lengkap (Semua Tahap Sekaligus)

```python
# Jalankan semua tahap preprocessing sekaligus
stats = run_full_preprocessing(visualize_samples=True)

# Output:
# - Progress bar untuk setiap tahap dengan warna berbeda
# - Statistik lengkap per tahap
# - File preprocessed_data.npz berisi data yang siap untuk training
# - Folder final_images berisi semua gambar hasil preprocessing
```

### 3️⃣ Jalankan Per Tahap (Testing/Debugging)

```python
from pathlib import Path

# TAHAP 1: Denoising
print("Menjalankan Denoising...")
stats_denoise = process_denoising(
    input_dir=Path("dataset/original_dataset"),
    output_dir=Path("dataset/output_denoised")
)

# TAHAP 2: Resizing
print("Menjalankan Resizing...")
stats_resize = process_resizing(
    input_dir=Path("dataset/output_denoised"),
    output_dir=Path("dataset/output_resized")
)

# TAHAP 3: Enhancement (CLAHE)
print("Menjalankan Enhancement...")
stats_enhance = process_enhancement(
    input_dir=Path("dataset/output_resized"),
    output_dir=Path("dataset/output_enhanced")
)

# TAHAP 4: Augmentation
print("Menjalankan Augmentation...")
stats_augment = process_augmentation(
    input_dir=Path("dataset/output_enhanced"),
    output_dir=Path("dataset/output_augmented")
)

# TAHAP 5: Normalization & Save
print("Menjalankan Normalization...")
stats_norm = save_normalized_data(
    input_dir=Path("dataset/output_augmented"),
    output_file=Path("dataset/preprocessed_data.npz")
)
```

### 4️⃣ Load Data yang Sudah Dipreprocess

```python
import numpy as np

# Load data dari file .npz
data = np.load('dataset/preprocessed_dataset/preprocessed_data.npz', allow_pickle=True)

# Extract data
X = data['X']  # Gambar yang sudah dinormalisasi (N, 224, 224, 3)
y = data['y']  # Label (N,)
class_names = data['class_names']  # Nama-nama kelas
class_to_idx = data['class_to_idx'].item()  # Mapping kelas ke index

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Number of classes: {len(class_names)}")
print(f"Class names: {class_names}")
```

### 5️⃣ Visualisasi Sample Gambar

```python
import matplotlib.pyplot as plt
import cv2

# Visualisasi dari satu fungsi
visualize_preprocessing_samples(
    processed_dir=Path("dataset/preprocessed_dataset/final_images"),
    num_samples=5
)

# ATAU manual
def show_samples(num_samples=10):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    # Ambil random samples
    indices = np.random.choice(len(X), num_samples, replace=False)

    for i, idx in enumerate(indices):
        # Denormalisasi untuk visualisasi
        img = X[idx]

        # Reverse normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_denorm = (img * std + mean) * 255
        img_denorm = np.clip(img_denorm, 0, 255).astype(np.uint8)

        # Tampilkan
        axes[i].imshow(img_denorm)
        axes[i].set_title(f"Class: {class_names[y[idx]]}", fontsize=8)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

show_samples()
```

### 6️⃣ Cek Distribusi Data Hasil Augmentasi

```python
from collections import Counter

# Hitung distribusi kelas
class_distribution = Counter(y)

# Tampilkan
print("Distribusi Data per Kelas:")
print("-" * 50)
for class_idx, count in sorted(class_distribution.items()):
    class_name = class_names[class_idx]
    print(f"{class_name}: {count} gambar")

# Visualisasi
plt.figure(figsize=(12, 6))
plt.bar(range(len(class_distribution)),
        [class_distribution[i] for i in range(len(class_names))],
        color='steelblue')
plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
plt.xlabel('Kelas')
plt.ylabel('Jumlah Gambar')
plt.title('Distribusi Dataset Setelah Preprocessing & Augmentasi')
plt.tight_layout()
plt.show()
```

---

## 🎯 Fitur-Fitur Utama

### ✨ Progress Bar dengan Warna

Setiap tahap memiliki progress bar dengan warna berbeda:

- 🔵 **Denoising** - Biru
- 🟢 **Resizing** - Hijau
- 🟡 **Enhancement** - Kuning
- 🟣 **Augmentation** - Magenta
- 🔵 **Normalization** - Cyan

### 📊 Statistik Real-time

Progress bar menampilkan:

- Persentase completion
- Kecepatan processing (img/s)
- Estimasi waktu tersisa
- Jumlah gambar yang diproses

### 🔄 Augmentasi yang Diterapkan

1. Flip Horizontal
2. Rotasi 90°
3. Brightness Increase
4. Brightness Decrease

### 📦 Output Files

- `preprocessed_data.npz` - Data tensor siap training
- `final_images/` - Folder gambar hasil preprocessing
- `sample_visualization.png` - Preview hasil

---

## ⚙️ Kustomisasi Konfigurasi

```python
# Ubah ukuran target
Config.TARGET_SIZE = (256, 256)

# Nonaktifkan augmentasi (untuk testing)
Config.AUGMENTATION_ENABLED = False

# Ubah parameter CLAHE
Config.CLAHE_CLIP_LIMIT = 3.0
Config.CLAHE_TILE_SIZE = (16, 16)

# Ubah parameter denoising
Config.DENOISE_H = 15
Config.DENOISE_H_COLOR = 15

# Path custom
Config.INPUT_DIR = Path("path/ke/dataset/anda")
Config.OUTPUT_DIR = Path("path/ke/output")
```

---

## 🐛 Troubleshooting

### Error: Module not found

```bash
# Install dependencies
pip install opencv-python tqdm pillow numpy pandas matplotlib
```

### Memory Error

Jika dataset terlalu besar, proses per batch:

```python
# Proses secara bertahap, jangan load semua sekaligus
# Atau kurangi ukuran target
Config.TARGET_SIZE = (128, 128)
```

### Progress bar tidak muncul

Pastikan menggunakan `tqdm.auto`:

```python
from tqdm.auto import tqdm
```

---

## 📝 Catatan Penting

1. **Backup Data Original**: Script ini tidak mengubah data original, semua output disimpan di folder terpisah
2. **Disk Space**: Augmentasi akan meningkatkan ukuran dataset 4-5x lipat
3. **Processing Time**: Tergantung ukuran dataset, bisa memakan waktu 30 menit - 2 jam
4. **RAM Usage**: Normalization step membutuhkan RAM yang cukup besar (minimum 8GB untuk dataset besar)

---

## 🎉 Selamat Preprocessing!

Jika ada error atau pertanyaan, silakan cek dokumentasi atau tanyakan! 😊
