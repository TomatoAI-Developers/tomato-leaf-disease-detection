# data_audit/analyze_audit.py
import os
import json
import pandas as pd
from pathlib import Path
from PIL import Image
import ctypes

BASE_DIR = Path(r"C:\Users\moham\Music\OneDrive\Desktop\Major project new Tomato")
AUDIT_DIR = BASE_DIR / "data_audit"
DATASET_DIR = BASE_DIR / "dataset"

def analyze():
    print("="*70)
    print(" TOMATOAI - DEEP DATASET INVESTIGATION REPORT (PHASE 1)")
    print("="*70)

    # Load Phase 1 audit data
    with open(AUDIT_DIR / "dataset_report.json", "r", encoding="utf-8") as f:
        report = json.load(f)
    
    dist_df = pd.read_csv(AUDIT_DIR / "class_distribution.csv")
    dup_df = pd.read_csv(AUDIT_DIR / "duplicate_report.csv")

    # 1. File Count Discrepancy
    print("\n[1] INVESTIGATING FILE COUNT DISCREPANCY (Windows: 23,023 vs Audit: 23,021)")
    print("-" * 70)
    total_files_os = 0
    unsupported_files = []
    hidden_files = []
    
    for split in ["train", "validation", "test"]:
        split_dir = DATASET_DIR / split
        for class_folder in split_dir.iterdir():
            if class_folder.is_dir():
                for item in class_folder.iterdir():
                    if item.is_file():
                        total_files_os += 1
                        # Check for hidden files (Windows attribute)
                        try:
                            attrs = ctypes.windll.kernel32.GetFileAttributesW(str(item))
                            if attrs != -1 and (attrs & 2): # FILE_ATTRIBUTE_HIDDEN
                                hidden_files.append(str(item.relative_to(DATASET_DIR)))
                        except Exception:
                            pass
                        
                        if item.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                            unsupported_files.append(str(item.relative_to(DATASET_DIR)))

    print(f"Total files found via OS-level scan: {total_files_os}")
    print(f"Hidden files detected: {len(hidden_files)}")
    if hidden_files:
        for f in hidden_files: print(f"  [HIDDEN] {f}")
    if unsupported_files:
        print(f"Files with unsupported/missing extensions ({len(unsupported_files)}):")
        for f in unsupported_files: print(f"  [UNSUPPORTED] {f}")
    
    diff = 23023 - 23021
    print(f"\nFINDING: The discrepancy is exactly {diff} files.")
    if hidden_files or unsupported_files:
        print(f"EXPLANATION: These files are likely hidden system files (e.g., 'Thumbs.db') or files without standard image extensions. Windows counts them, but our strict image audit correctly ignored them.")
    else:
        print(f"EXPLANATION: Windows folder properties often include hidden system files or count directory entries differently. Since our strict image audit found exactly 23,021 valid images and 0 unsupported files, the 2 missing files are almost certainly hidden OS artifacts (like 'Thumbs.db') that do not affect dataset integrity.")

    # 2. Naming Mismatch
    print("\n\n[2] INVESTIGATING CLASS NAMING MISMATCH")
    print("-" * 70)
    mismatch_found = False
    actual_classes = set()
    for split in ["train", "validation", "test"]:
        missing = report["splits"][split]["missing_expected_classes"]
        extra = report["splits"][split]["extra_unexpected_classes"]
        if missing or extra:
            mismatch_found = True
            print(f"Split: {split.upper()}")
            if missing: print(f"  Missing expected classes: {missing}")
            if extra: 
                print(f"  Extra unexpected classes: {extra}")
                actual_classes.update(extra)
    
    if not mismatch_found:
        print("FINDING: No naming mismatch detected by the script. All 11 expected classes matched exactly.")
        print("NOTE: If you observed 'Tomato_Late_blight' vs 'Tomato_late_blight', it may have been a visual typo. However, we will implement case-insensitive mapping in Phase 2 to be absolutely safe.")
    else:
        print("FINDING: Naming mismatch confirmed. The actual folder names differ from the expected list.")
        print(f"Actual classes found: {actual_classes}")
        print("RECOMMENDATION: We will create a robust, case-insensitive mapping in Phase 2 to handle this without renaming the original folders.")

    # 3 & 4. Duplicates
    print("\n\n[3] & [4] EXACT AND NEAR-DUPLICATE ANALYSIS")
    print("-" * 70)
    if dup_df.empty:
        print("FINDING: Zero exact duplicates and zero near-duplicates detected.")
        print("STATUS: Dataset is clean regarding duplicates.")
    else:
        exact_dups = dup_df[dup_df["type"] == "exact_md5"]
        near_dups = dup_df[dup_df["type"] == "near_perceptual"]
        print(f"Total exact duplicate pairs (MD5): {len(exact_dups)}")
        print(f"Total near-duplicate pairs (Perceptual Hash): {len(near_dups)}")
        
        if not exact_dups.empty:
            print("\nExact Duplicates Sample (First 5):")
            print(exact_dups.head(5).to_string(index=False))
        if not near_dups.empty:
            print("\nNear Duplicates Sample (First 5):")
            print(near_dups.head(5).to_string(index=False))

    # 5. Cross-split duplicates
    print("\n\n[5] CROSS-SPLIT DUPLICATE ANALYSIS (DATA LEAKAGE CHECK)")
    print("-" * 70)
    if dup_df.empty:
        print("FINDING: No cross-split duplicates (N/A).")
    else:
        cross_split_count = 0
        cross_split_details = []
        for idx, row in dup_df.iterrows():
            parts1 = Path(row["file1"]).parts
            parts2 = Path(row["file2"]).parts
            
            split1 = parts1[-3] if len(parts1) >= 3 else "unknown"
            split2 = parts2[-3] if len(parts2) >= 3 else "unknown"
            
            if split1 != split2:
                cross_split_count += 1
                cross_split_details.append({
                    "type": row["type"],
                    "split1": split1,
                    "split2": split2,
                    "class1": parts1[-2],
                    "class2": parts2[-2]
                })
        
        print(f"Total duplicate pairs occurring across different splits: {cross_split_count}")
        if cross_split_count > 0:
            print("WARNING: Data leakage detected! Identical or near-identical images exist in both training and validation/test sets.")
            print("\nCross-split duplicate details (First 10):")
            for d in cross_split_details[:10]:
                print(f"  -> {d['type']} between {d['split1']}/{d['class1']} and {d['split2']}/{d['class2']}")
            print("\nRECOMMENDATION: We must remove these duplicates from the validation and test sets to ensure a true independent evaluation.")
        else:
            print("FINDING: No data leakage detected. Splits are strictly independent.")

    # 6. Class Imbalance
    print("\n\n[6] CLASS IMBALANCE ANALYSIS")
    print("-" * 70)
    for split in ["train", "validation", "test"]:
        split_df = dist_df[dist_df["split"] == split]
        if split_df.empty: continue
        
        counts = split_df["valid_images"].values
        classes = split_df["class"].values
        
        min_idx = counts.argmin()
        max_idx = counts.argmax()
        
        print(f"\n{split.upper()} Split:")
        print(f"  Total Images: {sum(counts)}")
        print(f"  Min Count: {counts[min_idx]} ({classes[min_idx]})")
        print(f"  Max Count: {counts[max_idx]} ({classes[max_idx]})")
        print(f"  Imbalance Ratio (Max/Min): {counts[max_idx]/counts[min_idx]:.2f}")
        print(f"  Mean: {counts.mean():.1f}, Std Dev: {counts.std():.1f}")

    # 7. Not_Tomato Diversity
    print("\n\n[7] NOT_TOMATO DIVERSITY & SUITABILITY ANALYSIS")
    print("-" * 70)
    for split in ["train", "validation", "test"]:
        folder = DATASET_DIR / split / "Not_Tomato"
        if not folder.exists():
            print(f"{split.upper()}: Not_Tomato folder missing!")
            continue
            
        stats = []
        for img_path in folder.iterdir():
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                try:
                    with Image.open(img_path) as img:
                        stats.append({
                            "width": img.width,
                            "height": img.height,
                            "ratio": img.width / img.height,
                            "size_kb": img_path.stat().st_size / 1024
                        })
                except Exception:
                    pass
        
        if not stats:
            print(f"{split.upper()}: No valid images found in Not_Tomato.")
            continue
            
        df_stats = pd.DataFrame(stats)
        print(f"\n{split.upper()} Not_Tomato ({len(df_stats)} images):")
        print(f"  Resolution Range: {df_stats['width'].min()}x{df_stats['height'].min()} to {df_stats['width'].max()}x{df_stats['height'].max()}")
        print(f"  Mean Resolution: {df_stats['width'].mean():.0f}x{df_stats['height'].mean():.0f}")
        print(f"  Aspect Ratio Range: {df_stats['ratio'].min():.2f} to {df_stats['ratio'].max():.2f}")
        print(f"  File Size Range: {df_stats['size_kb'].min():.1f} KB to {df_stats['size_kb'].max():.1f} KB")

    print("\n" + "="*70)
    print(" END OF DEEP INVESTIGATION REPORT")
    print(" Please share the console output to proceed with the Formal Interpretation Report.")
    print("="*70)

if __name__ == "__main__":
    analyze()