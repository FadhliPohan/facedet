#!/usr/bin/env python3
"""
Script untuk menampilkan distribusi class di folder final_images
Tanpa loading data, hanya menghitung file untuk menghindari masalah memory
"""

import os
from pathlib import Path
from collections import Counter
import pandas as pd

def count_images_in_folder(folder_path):
    """Hitung jumlah gambar per class"""
    
    class_distribution = {}
    
    # Iterate through class folders
    for class_folder in sorted(os.listdir(folder_path)):
        class_path = os.path.join(folder_path, class_folder)
        
        if os.path.isdir(class_path):
            # Count image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            count = 0
            
            for file in os.listdir(class_path):
                file_path = os.path.join(class_path, file)
                if os.path.isfile(file_path):
                    ext = os.path.splitext(file)[1].lower()
                    if ext in image_extensions:
                        count += 1
            
            class_distribution[class_folder] = count
    
    return class_distribution


def display_distribution_table(distribution):
    """Tampilkan distribusi dalam bentuk tabel"""
    
    total = sum(distribution.values())
    
    print("\n" + "="*100)
    print("📊 DISTRIBUSI CLASS - FINAL IMAGES")
    print("="*100)
    
    # Create data for table
    data = []
    for idx, (class_name, count) in enumerate(sorted(distribution.items()), 1):
        percentage = (count / total * 100) if total > 0 else 0
        data.append({
            'No': idx,
            'Class Name': class_name,
            'Count': f'{count:,}',
            'Percentage': f'{percentage:.2f}%'
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Display table
    print("\n" + df.to_string(index=False))
    
    # Summary
    print("\n" + "-"*100)
    print(f"{'TOTAL':<50} {total:>15,} {100.0:>15.2f}%")
    print("="*100)
    
    # Check balance
    counts = list(distribution.values())
    unique_counts = set(counts)
    
    if len(unique_counts) == 1:
        print(f"\n✅ Dataset BALANCED - setiap class memiliki {list(unique_counts)[0]:,} gambar")
    else:
        min_count = min(counts)
        max_count = max(counts)
        imbalance_ratio = max_count / min_count
        print(f"\n⚠️  Dataset IMBALANCED")
        print(f"   Minimum: {min_count:,} images")
        print(f"   Maximum: {max_count:,} images")
        print(f"   Imbalance ratio: {imbalance_ratio:.2f}:1")
    
    print("\n" + "="*100)
    
    return df


def main():
    """Main function"""
    
    final_images_dir = Path("dataset/praproses_result/final_images")
    
    if not final_images_dir.exists():
        print(f"❌ Folder tidak ditemukan: {final_images_dir}")
        return
    
    print(f"\n📂 Scanning folder: {final_images_dir}")
    
    # Count images
    distribution = count_images_in_folder(final_images_dir)
    
    # Display table
    df = display_distribution_table(distribution)
    
    # Save to CSV
    output_csv = final_images_dir.parent / "class_distribution.csv"
    df.to_csv(output_csv, index=False)
    print(f"\n💾 Distribusi juga disimpan ke: {output_csv}")


if __name__ == "__main__":
    main()
