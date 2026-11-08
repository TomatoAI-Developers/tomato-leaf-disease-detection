# data_audit/audit_dataset.py

import os
import json
import hashlib
import time
from pathlib import Path
from PIL import Image
import imagehash
import pandas as pd
from collections import defaultdict

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(r"C:\Users\moham\Music\OneDrive\Desktop\Major project new Tomato")
DATASET_DIR = BASE_DIR / "dataset"
AUDIT_DIR = BASE_DIR / "data_audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "validation", "test"]
EXPECTED_CLASSES = [
    "Not_Tomato", "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Healthy",
    "Tomato_late_blight", "Tomato_Leaf_Mold", "Tomato_Mosaic_Virus", 
    "Tomato_Septoria_Leaf_Spot", "Tomato_Spider_Mites", "Tomato_Target_Spot", 
    "Tomato_Yellow_Leaf_CurlVirus"
]

# Standardized mapping for the application
CLASS_MAPPING = {
    "Not_Tomato": "Not Tomato",
    "Tomato_Bacterial_spot": "Bacterial Spot",
    "Tomato_Early_blight": "Early Blight",
    "Tomato_Healthy": "Healthy",
    "Tomato_late_blight": "Late Blight",
    "Tomato_Leaf_Mold": "Leaf Mold",
    "Tomato_Mosaic_Virus": "Mosaic Virus",
    "Tomato_Septoria_Leaf_Spot": "Septoria Leaf Spot",
    "Tomato_Spider_Mites": "Spider Mites",
    "Tomato_Target_Spot": "Target Spot",
    "Tomato_Yellow_Leaf_CurlVirus": "Yellow Leaf Curl Virus"
}

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def get_file_md5(filepath):
    """Calculate MD5 hash for exact duplicate detection."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_image_phash(filepath):
    """Calculate perceptual hash for near-duplicate detection."""
    try:
        with Image.open(filepath) as img:
            return str(imagehash.phash(img))
    except Exception:
        return None

# ==========================================
# MAIN AUDIT LOGIC
# ==========================================
def audit_dataset():
    print("="*60)
    print("STARTING TOMATOAI DATASET AUDIT (PHASE 1)")
    print("="*60)
    start_time = time.time()
    
    class_distribution = []
    invalid_files_log = []
    exact_duplicates_map = defaultdict(list)
    near_duplicates_map = defaultdict(list)
    
    split_class_counts = {split: defaultdict(int) for split in SPLITS}
    split_total_files = {split: defaultdict(int) for split in SPLITS}
    missing_classes = {split: [] for split in SPLITS}
    extra_classes = {split: [] for split in SPLITS}
    
    total_valid = 0
    total_invalid = 0
    total_unsupported = 0
    
    for split in SPLITS:
        split_dir = DATASET_DIR / split
        print(f"\n[INFO] Auditing split: {split.upper()}")
        
        if not split_dir.exists():
            print(f"  [ERROR] Split directory {split_dir} does not exist!")
            continue
            
        found_classes = set()
        
        for class_folder in sorted(split_dir.iterdir()):
            if not class_folder.is_dir():
                continue
                
            class_name = class_folder.name
            found_classes.add(class_name)
            
            if class_name not in EXPECTED_CLASSES:
                extra_classes[split].append(class_name)
                print(f"  [WARNING] Unexpected class folder found: {class_name}")
                
            total_files = 0
            valid_count = 0
            invalid_count = 0
            unsupported_count = 0
            
            for file_path in class_folder.iterdir():
                if not file_path.is_file():
                    continue
                total_files += 1
                
                ext = file_path.suffix.lower()
                if ext not in SUPPORTED_EXTENSIONS:
                    unsupported_count += 1
                    total_unsupported += 1
                    invalid_files_log.append({
                        "split": split, "class": class_name, 
                        "file": file_path.name, "reason": "Unsupported format"
                    })
                    continue
                
                # Verify image integrity
                try:
                    with Image.open(file_path) as img:
                        img.verify()
                    
                    valid_count += 1
                    total_valid += 1
                    
                    # Hash calculations for duplicate detection
                    md5_hash = get_file_md5(file_path)
                    exact_duplicates_map[md5_hash].append(str(file_path))
                    
                    phash = get_image_phash(file_path)
                    if phash:
                        near_duplicates_map[phash].append(str(file_path))
                        
                except Exception as e:
                    invalid_count += 1
                    total_invalid += 1
                    invalid_files_log.append({
                        "split": split, "class": class_name, 
                        "file": file_path.name, "reason": f"Corrupted/Unreadable"
                    })
            
            # Store counts
            split_class_counts[split][class_name] = valid_count
            split_total_files[split][class_name] = total_files
            
            class_distribution.append({
                "split": split, "class": class_name,
                "total_files_in_folder": total_files, # Explains Windows properties
                "valid_images": valid_count,
                "invalid_images": invalid_count,
                "unsupported_files": unsupported_count
            })
            print(f"  [OK] {class_name}: {valid_count} valid | {invalid_count} corrupt | {unsupported_count} unsupported")
            
        # Check for missing classes in this split
        for expected in EXPECTED_CLASSES:
            if expected not in found_classes:
                missing_classes[split].append(expected)
                
    # ==========================================
    # PROCESS DUPLICATES
    # ==========================================
    exact_dup_list = []
    for h, files in exact_duplicates_map.items():
        if len(files) > 1:
            for i in range(len(files)):
                for j in range(i+1, len(files)):
                    exact_dup_list.append({"hash": h, "file1": files[i], "file2": files[j], "type": "exact_md5"})
                    
    near_dup_list = []
    for h, files in near_duplicates_map.items():
        if len(files) > 1:
            for i in range(len(files)):
                for j in range(i+1, len(files)):
                    # Exclude if already counted as exact duplicate
                    if not any(d["file1"] == files[i] and d["file2"] == files[j] for d in exact_dup_list):
                        near_dup_list.append({"hash": h, "file1": files[i], "file2": files[j], "type": "near_perceptual"})

    # ==========================================
    # GENERATE REPORTS
    # ==========================================
    print("\n[INFO] Generating audit reports...")
    
    # 1. JSON Summary Report
    report = {
        "audit_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_path": str(DATASET_DIR),
        "totals": {
            "valid_images": total_valid,
            "invalid_corrupted": total_invalid,
            "unsupported_formats": total_unsupported,
            "exact_duplicates": len(exact_dup_list),
            "near_duplicates": len(near_dup_list)
        },
        "splits": {}
    }
    
    for split in SPLITS:
        report["splits"][split] = {
            "total_valid_images": sum(split_class_counts[split].values()),
            "total_folder_files": sum(split_total_files[split].values()),
            "classes_found": list(split_class_counts[split].keys()),
            "missing_expected_classes": missing_classes[split],
            "extra_unexpected_classes": extra_classes[split],
            "class_valid_counts": dict(split_class_counts[split]),
            "class_total_file_counts": dict(split_total_files[split])
        }
        
    with open(AUDIT_DIR / "dataset_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
        
    # 2. CSV Reports
    pd.DataFrame(class_distribution).to_csv(AUDIT_DIR / "class_distribution.csv", index=False)
    pd.DataFrame(invalid_files_log).to_csv(AUDIT_DIR / "invalid_files.csv", index=False)
    
    all_dups = exact_dup_list + near_dup_list
    if all_dups:
        pd.DataFrame(all_dups).to_csv(AUDIT_DIR / "duplicate_report.csv", index=False)
    else:
        pd.DataFrame(columns=["hash", "file1", "file2", "type"]).to_csv(AUDIT_DIR / "duplicate_report.csv", index=False)
        
    # 3. Class Mapping
    with open(AUDIT_DIR / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump(CLASS_MAPPING, f, indent=4)
        
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print(f"AUDIT COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS.")
    print(f"Reports saved to: {AUDIT_DIR}")
    print("="*60)

if __name__ == "__main__":
    audit_dataset()