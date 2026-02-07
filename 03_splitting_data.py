# -*- coding: utf-8 -*-
"""
========================================================================================
SCRIPT SPLITTING DATA - TRAIN/VALIDATION/TEST
========================================================================================
Split dataset hasil preprocessing menjadi:
- 70% Training
- 15% Validation
- 15% Test

Menggunakan stratified split untuk mempertahankan proporsi kelas di setiap split.
========================================================================================
"""

import os
import shutil
import json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt

# Konfigurasi
input_dir = "dataset/praproses_result/final_images"
output_dir = "dataset/split_data"

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed untuk reproducibility
RANDOM_SEED = 42


def validate_ratios():
    """Validasi bahwa total ratio = 1.0"""
    total = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    if not np.isclose(total, 1.0):
        raise ValueError(f"Total ratio harus = 1.0, saat ini: {total}")
    print(f"✅ Ratio validation passed: {TRAIN_RATIO:.0%} + {VAL_RATIO:.0%} + {TEST_RATIO:.0%} = {total:.0%}")


def collect_all_images(input_path):
    """
    Kumpulkan semua gambar dengan label kelasnya
    
    Returns:
        all_images: list of tuples (image_path, class_name, class_idx)
        class_names: list of class names
    """
    print("\n" + "="*80)
    print("📂 COLLECTING IMAGES FROM INPUT DIRECTORY")
    print("="*80)
    
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory tidak ditemukan: {input_path}")
    
    # Get all class folders
    class_folders = sorted([f for f in input_path.iterdir() if f.is_dir()])
    
    if len(class_folders) == 0:
        raise ValueError(f"Tidak ada folder kelas ditemukan di: {input_path}")
    
    class_names = [f.name for f in class_folders]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    print(f"\n📊 Found {len(class_names)} classes:")
    for idx, name in enumerate(class_names):
        print(f"   {idx}: {name}")
    
    # Collect all images
    all_images = []
    class_counts = Counter()
    
    print("\n📂 Processing classes...")
    for idx, class_folder in enumerate(class_folders, 1):
        class_name = class_folder.name
        class_idx = class_to_idx[class_name]
        print(f"   [{idx}/{len(class_folders)}] {class_name}")
        
        # Get all image files
        image_files = list(class_folder.glob('*.jpg')) + \
                      list(class_folder.glob('*.jpeg')) + \
                      list(class_folder.glob('*.png')) + \
                      list(class_folder.glob('*.bmp'))
        
        for img_path in image_files:
            all_images.append((img_path, class_name, class_idx))
            class_counts[class_name] += 1
    
    print(f"\n📊 Class distribution:")
    print(f"   {'Class':<50} {'Count':>10}")
    print("   " + "-"*62)
    for class_name in sorted(class_counts.keys()):
        count = class_counts[class_name]
        print(f"   {class_name:<50} {count:>10,}")
    print("   " + "-"*62)
    print(f"   {'TOTAL':<50} {len(all_images):>10,}")
    
    return all_images, class_names


def stratified_train_val_test_split(all_images, train_ratio, val_ratio, test_ratio, random_seed=42):
    """
    Stratified split: train/val/test dengan mempertahankan proporsi kelas
    
    Args:
        all_images: list of (path, class_name, class_idx)
        train_ratio: proporsi training
        val_ratio: proporsi validation
        test_ratio: proporsi testing
        random_seed: seed untuk reproducibility
    
    Returns:
        train_data, val_data, test_data: masing-masing list of (path, class_name, class_idx)
    """
    print("\n" + "="*80)
    print("✂️  STRATIFIED SPLIT - TRAIN/VAL/TEST")
    print("="*80)
    print(f"   Train: {train_ratio:.0%}")
    print(f"   Val:   {val_ratio:.0%}")
    print(f"   Test:  {test_ratio:.0%}")
    print(f"   Random seed: {random_seed}")
    
    # Group by class
    images_by_class = defaultdict(list)
    for img_path, class_name, class_idx in all_images:
        images_by_class[class_name].append((img_path, class_name, class_idx))
    
    train_data = []
    val_data = []
    test_data = []
    
    # Split per class untuk mempertahankan distribusi
    np.random.seed(random_seed)
    
    for class_name, images in images_by_class.items():
        n_images = len(images)
        
        # Shuffle images dalam kelas ini
        np.random.shuffle(images)
        
        # Hitung jumlah untuk setiap split
        n_train = int(n_images * train_ratio)
        n_val = int(n_images * val_ratio)
        # Sisa masuk ke test untuk memastikan tidak ada yang terbuang
        
        # Split
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        train_data.extend(train_images)
        val_data.extend(val_images)
        test_data.extend(test_images)
        
        print(f"   {class_name[:45]:<45}: {len(train_images):>5} train, {len(val_images):>5} val, {len(test_images):>5} test")
    
    # Shuffle final datasets
    np.random.shuffle(train_data)
    np.random.shuffle(val_data)
    np.random.shuffle(test_data)
    
    print(f"\n✅ Split complete:")
    print(f"   Train: {len(train_data):>6,} images ({len(train_data)/len(all_images):.1%})")
    print(f"   Val:   {len(val_data):>6,} images ({len(val_data)/len(all_images):.1%})")
    print(f"   Test:  {len(test_data):>6,} images ({len(test_data)/len(all_images):.1%})")
    print(f"   Total: {len(train_data) + len(val_data) + len(test_data):>6,} images")
    
    return train_data, val_data, test_data


def copy_images_to_split(data, split_name, output_path, use_symlink=False):
    """
    Copy (atau symlink) gambar ke folder split
    
    Args:
        data: list of (path, class_name, class_idx)
        split_name: 'train', 'val', atau 'test'
        output_path: base output directory
        use_symlink: jika True, gunakan symlink daripada copy (hemat space)
    """
    print(f"\n📁 Copying images to {split_name}...")
    
    split_dir = output_path / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    
    # Create class folders
    class_names = set([class_name for _, class_name, _ in data])
    for class_name in class_names:
        class_dir = split_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy/symlink files
    total = len(data)
    for idx, (img_path, class_name, _) in enumerate(data, 1):
        if idx % 1000 == 0 or idx == total:
            print(f"   Progress: {idx}/{total} ({idx/total*100:.1f}%)")
        dest_dir = split_dir / class_name
        dest_path = dest_dir / img_path.name
        
        if use_symlink:
            # Create symlink (relative path for portability)
            if not dest_path.exists():
                os.symlink(img_path.resolve(), dest_path)
        else:
            # Copy file
            shutil.copy2(img_path, dest_path)
    
    print(f"   ✅ {len(data):,} images copied to {split_dir}")


def generate_split_report(train_data, val_data, test_data, class_names, output_path):
    """
    Generate comprehensive report dengan statistik dan visualisasi
    """
    print("\n" + "="*80)
    print("📊 GENERATING SPLIT REPORT")
    print("="*80)
    
    # Collect statistics
    def get_class_distribution(data):
        counter = Counter([class_name for _, class_name, _ in data])
        return {class_name: counter[class_name] for class_name in class_names}
    
    train_dist = get_class_distribution(train_data)
    val_dist = get_class_distribution(val_data)
    test_dist = get_class_distribution(test_data)
    
    total_images = len(train_data) + len(val_data) + len(test_data)
    
    # Create report dict
    report = {
        'total_images': total_images,
        'train': {
            'count': len(train_data),
            'percentage': len(train_data) / total_images * 100,
            'distribution': train_dist
        },
        'val': {
            'count': len(val_data),
            'percentage': len(val_data) / total_images * 100,
            'distribution': val_dist
        },
        'test': {
            'count': len(test_data),
            'percentage': len(test_data) / total_images * 100,
            'distribution': test_dist
        },
        'class_names': class_names,
        'num_classes': len(class_names),
        'split_ratios': {
            'train': TRAIN_RATIO,
            'val': VAL_RATIO,
            'test': TEST_RATIO
        },
        'random_seed': RANDOM_SEED
    }
    
    # Save JSON report
    report_file = output_path / 'split_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✅ JSON report saved: {report_file}")
    
    # Generate text report
    text_report_file = output_path / 'split_report.txt'
    with open(text_report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DATA SPLITTING REPORT - TRAIN/VALIDATION/TEST\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total Images: {total_images:,}\n")
        f.write(f"Number of Classes: {len(class_names)}\n")
        f.write(f"Random Seed: {RANDOM_SEED}\n\n")
        
        f.write("Split Configuration:\n")
        f.write(f"  Train:      {TRAIN_RATIO:>5.0%} ({len(train_data):>6,} images)\n")
        f.write(f"  Validation: {VAL_RATIO:>5.0%} ({len(val_data):>6,} images)\n")
        f.write(f"  Test:       {TEST_RATIO:>5.0%} ({len(test_data):>6,} images)\n\n")
        
        f.write("="*80 + "\n")
        f.write("CLASS DISTRIBUTION PER SPLIT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"{'Class':<50} {'Train':>10} {'Val':>10} {'Test':>10} {'Total':>10}\n")
        f.write("-"*92 + "\n")
        
        for class_name in sorted(class_names):
            train_count = train_dist[class_name]
            val_count = val_dist[class_name]
            test_count = test_dist[class_name]
            total_count = train_count + val_count + test_count
            
            f.write(f"{class_name:<50} {train_count:>10,} {val_count:>10,} {test_count:>10,} {total_count:>10,}\n")
        
        f.write("-"*92 + "\n")
        f.write(f"{'TOTAL':<50} {len(train_data):>10,} {len(val_data):>10,} {len(test_data):>10,} {total_images:>10,}\n")
        f.write(f"{'PERCENTAGE':<50} {len(train_data)/total_images*100:>9.1f}% {len(val_data)/total_images*100:>9.1f}% {len(test_data)/total_images*100:>9.1f}% {100.0:>9.1f}%\n")
    
    print(f"✅ Text report saved: {text_report_file}")
    
    return report


def visualize_split_distribution(report, output_path):
    """
    Buat visualisasi distribusi kelas per split
    """
    print("\n📊 Creating visualizations...")
    
    class_names = report['class_names']
    
    # Prepare data
    train_counts = [report['train']['distribution'][cn] for cn in class_names]
    val_counts = [report['val']['distribution'][cn] for cn in class_names]
    test_counts = [report['test']['distribution'][cn] for cn in class_names]
    
    # 1. Stacked bar chart
    fig, ax = plt.subplots(figsize=(16, 8))
    
    x = np.arange(len(class_names))
    width = 0.6
    
    p1 = ax.bar(x, train_counts, width, label='Train', color='#3498db', alpha=0.8)
    p2 = ax.bar(x, val_counts, width, bottom=train_counts, label='Validation', color='#2ecc71', alpha=0.8)
    p3 = ax.bar(x, test_counts, width, bottom=np.array(train_counts) + np.array(val_counts), 
                label='Test', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Disease Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
    ax.set_title('Dataset Split Distribution per Class', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    viz1_file = output_path / 'split_distribution_stacked.png'
    plt.savefig(viz1_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Stacked bar chart: {viz1_file}")
    
    # 2. Grouped bar chart
    fig, ax = plt.subplots(figsize=(16, 8))
    
    width = 0.25
    x = np.arange(len(class_names))
    
    ax.bar(x - width, train_counts, width, label='Train', color='#3498db', alpha=0.8)
    ax.bar(x, val_counts, width, label='Validation', color='#2ecc71', alpha=0.8)
    ax.bar(x + width, test_counts, width, label='Test', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Disease Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
    ax.set_title('Dataset Split Distribution - Grouped Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    viz2_file = output_path / 'split_distribution_grouped.png'
    plt.savefig(viz2_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Grouped bar chart: {viz2_file}")
    
    # 3. Pie chart - overall split
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sizes = [report['train']['count'], report['val']['count'], report['test']['count']]
    labels = [f"Train\n{report['train']['count']:,}\n({report['train']['percentage']:.1f}%)",
              f"Validation\n{report['val']['count']:,}\n({report['val']['percentage']:.1f}%)",
              f"Test\n{report['test']['count']:,}\n({report['test']['percentage']:.1f}%)"]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    explode = (0.05, 0.05, 0.05)
    
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='',
           shadow=True, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax.set_title(f'Overall Dataset Split\nTotal: {report["total_images"]:,} images', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    viz3_file = output_path / 'split_distribution_pie.png'
    plt.savefig(viz3_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Pie chart: {viz3_file}")


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("🚀 DATA SPLITTING - TRAIN/VALIDATION/TEST")
    print("="*80)
    
    # Validate ratios
    validate_ratios()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output directory: {output_path.resolve()}")
    
    # Step 1: Collect all images
    all_images, class_names = collect_all_images(input_dir)
    
    # Step 2: Stratified split
    train_data, val_data, test_data = stratified_train_val_test_split(
        all_images, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
    )
    
    # Step 3: Copy images to split folders
    copy_images_to_split(train_data, 'train', output_path, use_symlink=False)
    copy_images_to_split(val_data, 'val', output_path, use_symlink=False)
    copy_images_to_split(test_data, 'test', output_path, use_symlink=False)
    
    # Step 4: Generate report
    report = generate_split_report(train_data, val_data, test_data, class_names, output_path)
    
    # Step 5: Visualize
    visualize_split_distribution(report, output_path)
    
    # Final summary
    print("\n" + "="*80)
    print("✅ DATA SPLITTING COMPLETE!")
    print("="*80)
    print(f"📂 Output directory: {output_path.resolve()}")
    print(f"   📁 train/     : {len(train_data):>6,} images ({TRAIN_RATIO:.0%})")
    print(f"   📁 val/       : {len(val_data):>6,} images ({VAL_RATIO:.0%})")
    print(f"   📁 test/      : {len(test_data):>6,} images ({TEST_RATIO:.0%})")
    print(f"   📄 Reports    : split_report.json, split_report.txt")
    print(f"   📊 Visualizations generated")
    print("="*80)
    
    # Print summary table
    print("\n📊 FINAL SUMMARY:")
    print(f"   {'Split':<15} {'Images':>10} {'Percentage':>12}")
    print("   " + "-"*39)
    print(f"   {'Train':<15} {len(train_data):>10,} {len(train_data)/(len(train_data)+len(val_data)+len(test_data))*100:>11.1f}%")
    print(f"   {'Validation':<15} {len(val_data):>10,} {len(val_data)/(len(train_data)+len(val_data)+len(test_data))*100:>11.1f}%")
    print(f"   {'Test':<15} {len(test_data):>10,} {len(test_data)/(len(train_data)+len(val_data)+len(test_data))*100:>11.1f}%")
    print("   " + "-"*39)
    print(f"   {'TOTAL':<15} {len(train_data)+len(val_data)+len(test_data):>10,} {100.0:>11.1f}%")


if __name__ == "__main__":
    main()