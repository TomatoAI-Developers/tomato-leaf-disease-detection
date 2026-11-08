import tensorflow as tf
import matplotlib.pyplot as plt
import json
import csv
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime

from .dataset_loader import load_datasets
from .class_weights import get_all_weights
from .preprocess import IMG_SIZE, CLASS_NAMES, DISEASE_CLASS_NAMES, SUPPORTED_EXTENSIONS, load_and_preprocess_image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "phase2b"

# Ensure RUN_DIR is strictly unique and never overwrites an existing run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
base_run_dir = ARTIFACTS_DIR / f"run_{timestamp}"
RUN_DIR = base_run_dir
counter = 1
while RUN_DIR.exists():
    RUN_DIR = ARTIFACTS_DIR / f"run_{timestamp}_{counter}"
    counter += 1

def get_file_hash(filepath):
    """Memory-efficient chunked MD5 hashing."""
    h = hashlib.md5()
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def validate_dataset_structure():
    """Checks for exactly 11 folders, matching names, and >0 images in all splits."""
    for split in ["train", "validation", "test"]:
        split_dir = DATASET_ROOT / split
        folders = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
        if folders != CLASS_NAMES:
            raise ValueError(f"Folder mismatch in {split}. Expected {CLASS_NAMES}, got {folders}")
        
        for cls in CLASS_NAMES:
            cls_dir = split_dir / cls
            count = sum(1 for p in cls_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
            if count == 0:
                raise ValueError(f"Class '{cls}' in {split} has 0 supported images.")
                
    print("[PASS] Dataset structure: Exactly 11 matching folders with >0 images across all splits.")

def validate_representative_images():
    """Decodes and validates 33 representative images (11 classes * 3 splits)."""
    print("Validating representative images (33 total)...")
    for split in ["train", "validation", "test"]:
        for cls in CLASS_NAMES:
            cls_dir = DATASET_ROOT / split / cls
            img_path = next((p for p in cls_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS), None)
            if img_path is None:
                raise ValueError(f"No supported image found for {cls} in {split}")

            img_array = load_and_preprocess_image(img_path)
            assert img_array.shape == (224, 224, 3), f"Shape mismatch for {img_path}"
            assert img_array.dtype == np.float32, f"Dtype mismatch for {img_path}"
            assert img_array.min() >= 0.0 and img_array.max() <= 1.0, f"Range mismatch for {img_path}"
    print("[PASS] Representative images: 33 images decoded, shape (224,224,3), Float32, [0,1].")

def validate_data_pipeline(train_ds, val_ds, test_ds):
    """Validates image loading, shape, normalization, label ranges, and augmentation leakage."""
    # 1. Training Pipeline Smoke Test
    seen_labels = set()
    batch_count = 0
    images_checked = 0
    for images, labels in train_ds:
        assert images.shape[1:] == (224, 224, 3), "Train shape mismatch"
        assert images.dtype == tf.float32, "Train dtype mismatch"
        assert tf.reduce_min(images) >= 0.0 and tf.reduce_max(images) <= 1.0, "Train normalization failed"
        assert tf.reduce_min(labels) >= 0 and tf.reduce_max(labels) <= 10, "Train label range failed (0-10)"
        for lbl in labels.numpy():
            seen_labels.add(lbl)
        batch_count += 1
        images_checked += len(labels)
        if len(seen_labels) == 11:
            break
            
    if len(seen_labels) != 11:
        raise ValueError(f"Train pipeline smoke test failed to load images from all 11 classes. Found: {len(seen_labels)}")
    print(f"[PASS] Train pipeline smoke test: Checked {batch_count} batches ({images_checked} images). "
          f"Shape (224,224,3), Float32, Normalized [0,1], Labels 0-10, All 11 classes loaded.")

    # 2. Val/Test Pipelines (Strict Augmentation Leakage Check)
    for ds, name in [(val_ds, "Validation"), (test_ds, "Test")]:
        batch1 = next(iter(ds))
        batch2 = next(iter(ds))
        
        img1 = batch1[0][0].numpy()
        img2 = batch2[0][0].numpy()
        
        if not np.array_equal(img1, img2):
            raise ValueError(f"{name} pipeline is applying augmentation! Images are not deterministic.")
            
        images, labels = batch1 # Explicit assignment
        assert images.shape[1:] == (224, 224, 3), f"{name} shape mismatch"
        assert images.dtype == tf.float32, f"{name} dtype mismatch"
        assert tf.reduce_min(images) >= 0.0 and tf.reduce_max(images) <= 1.0, f"{name} normalization failed"
        assert tf.reduce_min(labels) >= 0 and tf.reduce_max(labels) <= 10, f"{name} label range failed"
        print(f"[PASS] {name} pipeline: Shape (224,224,3), Float32, Normalized [0,1], No Augmentation (Deterministic).")

def verify_duplicates_separate():
    """Accurate pairwise cross-split duplicate verification using chunked hashing."""
    print("Starting separate duplicate verification (MD5)...")
    hashes = {}
    
    for split in ["train", "validation", "test"]:
        split_dir = DATASET_ROOT / split
        for cls_dir in split_dir.iterdir():
            if cls_dir.is_dir():
                for img_path in cls_dir.iterdir():
                    if img_path.is_file() and img_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                        file_hash = get_file_hash(img_path)
                        if file_hash not in hashes:
                            hashes[file_hash] = []
                        hashes[file_hash].append(img_path)
                            
    cross_split_pairs = 0
    for file_hash, paths in hashes.items():
        if len(paths) > 1:
            for i in range(len(paths)):
                for j in range(i + 1, len(paths)):
                    if paths[i].parent.parent.name != paths[j].parent.parent.name:
                        cross_split_pairs += 1
                        
    return cross_split_pairs, len(hashes)

def generate_reports_and_mappings():
    """Generates CSV, JSON reports, and mappings strictly inside RUN_DIR."""
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR = RUN_DIR / "preprocessing_samples"
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Class Distribution CSV
    csv_path = RUN_DIR / "final_class_distribution.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Class", "Train", "Validation", "Test"])
        for cls in CLASS_NAMES:
            counts = []
            for split in ["train", "validation", "test"]:
                counts.append(sum(1 for p in (DATASET_ROOT / split / cls).iterdir() 
                                  if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS))
            writer.writerow([cls] + counts)
            
    # 2. Class Mappings (Saved in RUN_DIR, existing metadata untouched)
    mapping_11 = {cls: i for i, cls in enumerate(CLASS_NAMES)}
    mapping_10 = {cls: i for i, cls in enumerate(DISEASE_CLASS_NAMES)}
    
    # Strict mapping assertions
    assert len(mapping_11) == 11 and list(mapping_11.values()) == list(range(11)), "11-class mapping invalid"
    assert len(mapping_10) == 10 and list(mapping_10.values()) == list(range(10)), "10-class mapping invalid"
    assert "Not_Tomato" not in mapping_10, "Not_Tomato incorrectly included in disease mapping"
    
    with open(RUN_DIR / "class_mapping.json", "w") as f:
        json.dump(mapping_11, f, indent=4)
    with open(RUN_DIR / "disease_class_mapping.json", "w") as f:
        json.dump(mapping_10, f, indent=4)
        
    # 3. Preprocessing Report
    cross_pairs, unique_hashes = verify_duplicates_separate()
    report = {
        "timestamp": datetime.now().isoformat(),
        "image_size": list(IMG_SIZE),
        "batch_size": 16,
        "num_classes_11": len(CLASS_NAMES),
        "num_classes_10": len(DISEASE_CLASS_NAMES),
        "class_names_11": CLASS_NAMES,
        "class_names_10": DISEASE_CLASS_NAMES,
        "augmentation_applied_to": "train_only",
        "separate_duplicate_verification": {
            "cross_split_duplicate_pairs_found": cross_pairs,
            "unique_hashes_found": unique_hashes,
            "note": "Separate verification scan. Phase 1 results preserved in data_audit/."
        }
    }
    with open(RUN_DIR / "preprocessing_report.json", "w") as f:
        json.dump(report, f, indent=4)
        
    return SAMPLES_DIR

def save_visual_samples(train_ds, samples_dir):
    """Saves augmented samples to verify pipeline. Skips if file exists."""
    for images, labels in train_ds.take(1):
        for i in range(min(3, len(images))):
            img = images[i].numpy()
            label_idx = labels[i].numpy()
            cls_name = CLASS_NAMES[label_idx]
            filename = samples_dir / f"sample_{cls_name}_{i}.png"
            
            if filename.exists():
                print(f"[SKIP] Visual sample already exists: {filename}")
                continue
                
            plt.figure(figsize=(4, 4))
            plt.imshow(img)
            plt.title(f"Augmented: {cls_name}")
            plt.axis('off')
            plt.savefig(filename)
            plt.close()
    print(f"[PASS] Saved visual samples to {samples_dir}")

def main():
    print(f"--- Phase 2B Validation Started (Run Dir: {RUN_DIR.name}) ---")
    
    print("1. Validating dataset structure...")
    validate_dataset_structure()
    
    print("2. Validating representative images...")
    validate_representative_images()
    
    print("3. Loading datasets...")
    train_ds, val_ds, test_ds, class_names = load_datasets()
    
    print("4. Running training pipeline smoke test...")
    validate_data_pipeline(train_ds, val_ds, test_ds)
    
    print("5. Calculating class weights...")
    get_all_weights(RUN_DIR)
    
    print("6. Generating reports and mappings...")
    samples_dir = generate_reports_and_mappings()
    
    print("7. Saving visual samples...")
    save_visual_samples(train_ds, samples_dir)
    
    print(f"--- Phase 2B Validation Complete. Outputs saved to: {RUN_DIR} ---")

if __name__ == "__main__":
    main()