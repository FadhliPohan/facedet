# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set encoding untuk Windows console (hanya jika bukan Jupyter Notebook)
if sys.platform == 'win32' and hasattr(sys.stdout, 'buffer'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Set style untuk visualisasi
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Path ke dataset
DATASET_PATH = Path("dataset/original_dataset")

def load_dataset_info():
    """
    Mengimport dataset dan menghitung jumlah gambar di setiap folder
    """
    dataset_info = []
    
    # Mendapatkan semua folder kelas
    class_folders = [f for f in DATASET_PATH.iterdir() if f.is_dir()]
    
    print("=" * 80)
    print("INFORMASI DATASET")
    print("=" * 80)
    print(f"Path Dataset: {DATASET_PATH.absolute()}")
    print(f"Jumlah Kelas: {len(class_folders)}")
    print("-" * 80)
    
    # Menghitung jumlah file di setiap kelas
    for folder in sorted(class_folders):
        # Hitung jumlah file gambar
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        image_files = [f for f in folder.iterdir() 
                      if f.is_file() and f.suffix.lower() in image_extensions]
        
        count = len(image_files)
        class_name = folder.name
        
        dataset_info.append({
            'Kelas': class_name,
            'Jumlah Gambar': count,
            'Path': str(folder)
        })
        
        print(f"{class_name}: {count} gambar")
    
    # Buat DataFrame
    df = pd.DataFrame(dataset_info)
    
    # Statistik keseluruhan
    total_images = df['Jumlah Gambar'].sum()
    avg_images = df['Jumlah Gambar'].mean()
    min_images = df['Jumlah Gambar'].min()
    max_images = df['Jumlah Gambar'].max()
    
    print("-" * 80)
    print(f"Total Gambar: {total_images:,}")
    print(f"Rata-rata per Kelas: {avg_images:.2f}")
    print(f"Minimum: {min_images}")
    print(f"Maximum: {max_images}")
    print("=" * 80)
    
    return df

def visualize_distribution(df):
    """
    Membuat visualisasi penyebaran data
    """
    # Persiapan data untuk visualisasi
    # Bersihkan nama kelas untuk tampilan yang lebih baik
    df_sorted = df.sort_values('Jumlah Gambar', ascending=False).copy()
    
    # Buat singkatan nama kelas untuk sumbu X
    df_sorted['Kelas_Singkat'] = df_sorted['Kelas'].apply(
        lambda x: x.split('.')[1].strip() if '.' in x else x
    )
    
    # Create subplot dengan 2 baris
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    # Plot 1: Bar chart horizontal dengan warna gradient
    colors = plt.cm.viridis(range(len(df_sorted)))
    bars = axes[0].barh(range(len(df_sorted)), df_sorted['Jumlah Gambar'], color=colors)
    axes[0].set_yticks(range(len(df_sorted)))
    axes[0].set_yticklabels(df_sorted['Kelas_Singkat'], fontsize=10)
    axes[0].set_xlabel('Jumlah Gambar', fontsize=12, fontweight='bold')
    axes[0].set_title('Distribusi Dataset per Kelas (Bar Chart)', 
                     fontsize=14, fontweight='bold', pad=20)
    axes[0].grid(axis='x', alpha=0.3)
    
    # Tambahkan label nilai di setiap bar
    for i, (bar, count) in enumerate(zip(bars, df_sorted['Jumlah Gambar'])):
        axes[0].text(count + 100, i, f'{count:,}', 
                    va='center', fontsize=9, fontweight='bold')
    
    # Plot 2: Pie chart
    # Hanya tampilkan label untuk kelas dengan > 5% data
    total = df_sorted['Jumlah Gambar'].sum()
    percentages = (df_sorted['Jumlah Gambar'] / total) * 100
    
    def autopct_format(pct):
        return f'{pct:.1f}%' if pct > 5 else ''
    
    wedges, texts, autotexts = axes[1].pie(
        df_sorted['Jumlah Gambar'], 
        labels=df_sorted['Kelas_Singkat'],
        autopct=autopct_format,
        startangle=90,
        colors=colors,
        textprops={'fontsize': 9}
    )
    
    # Perbaiki tampilan teks
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    axes[1].set_title('Proporsi Dataset per Kelas (Pie Chart)', 
                     fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Simpan visualisasi
    output_path = "dataset_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualisasi disimpan di: {output_path}")
    
    plt.show()

def check_data_balance(df):
    """
    Mengecek keseimbangan dataset
    """
    print("\n" + "=" * 80)
    print("ANALISIS KESEIMBANGAN DATA")
    print("=" * 80)
    
    total = df['Jumlah Gambar'].sum()
    
    # Hitung persentase setiap kelas
    df_analysis = df.copy()
    df_analysis['Persentase (%)'] = (df_analysis['Jumlah Gambar'] / total * 100).round(2)
    df_analysis = df_analysis.sort_values('Jumlah Gambar', ascending=False)
    
    print("\nRingkasan Persentase:")
    print("-" * 80)
    for idx, row in df_analysis.iterrows():
        class_name = row['Kelas'].split('.')[1].strip() if '.' in row['Kelas'] else row['Kelas']
        print(f"{class_name:50s} {row['Jumlah Gambar']:6,} ({row['Persentase (%)']:5.2f}%)")
    
    # Deteksi imbalance
    max_pct = df_analysis['Persentase (%)'].max()
    min_pct = df_analysis['Persentase (%)'].min()
    ratio = max_pct / min_pct
    
    print("-" * 80)
    print(f"Rasio Imbalance: {ratio:.2f}:1")
    
    if ratio > 10:
        print("⚠️  Dataset SANGAT TIDAK SEIMBANG - Pertimbangkan teknik resampling")
    elif ratio > 5:
        print("⚠️  Dataset TIDAK SEIMBANG - Mungkin perlu augmentasi")
    elif ratio > 2:
        print("ℹ️  Dataset agak tidak seimbang - Masih dalam batas wajar")
    else:
        print("✓ Dataset relatif seimbang")
    
    print("=" * 80)
    
    return df_analysis

def main():
    """
    Fungsi utama untuk menjalankan semua analisis
    """
    # Cek apakah folder dataset ada
    if not DATASET_PATH.exists():
        print(f"❌ Error: Folder {DATASET_PATH} tidak ditemukan!")
        return
    
    # Load dan tampilkan info dataset
    df = load_dataset_info()
    
    # Analisis keseimbangan
    df_analysis = check_data_balance(df)
    
    # Visualisasi distribusi
    visualize_distribution(df)
    
    # Simpan ringkasan ke CSV
    output_csv = "dataset_summary.csv"
    df_analysis.to_csv(output_csv, index=False)
    print(f"\n✓ Ringkasan dataset disimpan di: {output_csv}")

if __name__ == "__main__":
    main()
