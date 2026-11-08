import os
from pathlib import Path

BASE_DIR = Path(r"C:\Users\moham\Music\OneDrive\Desktop\Major project new Tomato")

files_content = {
    "preprocessing/preprocess.py": r"""import numpy as np
from PIL import Image, ImageOps

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    try:
        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32)
        img_array = img_array / 255.0
        return img_array
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None
""",

    "preprocessing/augmentation.py": r"""import tensorflow as tf

def get_augmentation_layer():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.1)
    ], name="augmentation_layer")
""",

    "preprocessing/dataset_loader.py": r"""import tensorflow as tf
import json
from pathlib import Path
from preprocessing.augmentation import get_augmentation_layer

BASE_DIR = Path(r"C:\Users\moham\Music\OneDrive\Desktop\Major project new Tomato")
METADATA_DIR = BASE_DIR / "artifacts" / "model_metadata"

def load_datasets():
    config_path = METADATA_DIR / "preprocessing_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    target_size = tuple(config["target_size"])
    batch_size = config["batch_size"]
    
    # UPDATED: Use 'dataset' instead of 'dataset_cleaned'
    train_dir = BASE_DIR / "dataset" / "train"
    val_dir = BASE_DIR / "dataset" / "validation"
    test_dir = BASE_DIR / "dataset" / "test"
    
    with open(METADATA_DIR / "class_mapping.json", 'r') as f:
        class_mapping = json.load(f)
        
    # Get actual folder names sorted by their assigned index
    actual_classes = [k for k, v in sorted(class_mapping.items(), key=lambda item: item[1]['index'])]
    
    autotune = tf.data.AUTOTUNE
    
    # Pass actual folder names to ensure TF finds the directories
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=target_size, batch_size=batch_size,
        class_names=actual_classes, shuffle=True, seed=42
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, image_size=target_size, batch_size=batch_size,
        class_names=actual_classes, shuffle=False, seed=42
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, image_size=target_size, batch_size=batch_size,
        class_names=actual_classes, shuffle=False, seed=42
    )
    
    aug_layer = get_augmentation_layer()
    
    train_ds = train_ds.cache().map(
        lambda x, y: (aug_layer(x, training=True), y), 
        num_parallel_calls=autotune
    ).prefetch(autotune)
    
    val_ds = val_ds.cache().prefetch(autotune)
    test_ds = test_ds.cache().prefetch(autotune)
    
    return train_ds, val_ds, test_ds, actual_classes
""",

    "preprocessing/class_weights.py": r"""import json
from pathlib import Path

BASE_DIR = Path(r"C:\Users\moham\Music\OneDrive\Desktop\Major project new Tomato")

def calculate_class_weights():
    # UPDATED: Use 'dataset' instead of 'dataset_cleaned'
    train_dir = BASE_DIR / "dataset" / "train"
    metadata_dir = BASE_DIR / "artifacts" / "model_metadata"
    
    with open(metadata_dir / "class_mapping.json", 'r') as f:
        class_mapping = json.load(f)
        
    actual_classes = [k for k, v in sorted(class_mapping.items(), key=lambda item: item[1]['index'])]
    
    counts = {}
    for class_name in actual_classes:
        class_dir = train_dir / class_name
        counts[class_name] = len(list(class_dir.glob("*"))) if class_dir.exists() else 0
            
    total_samples = sum(counts.values())
    num_classes = len(actual_classes)
    
    weights = {}
    for class_name in actual_classes:
        if counts[class_name] > 0:
            weights[class_name] = total_samples / (num_classes * counts[class_name])
        else:
            weights[class_name] = 0.0
            
    index_weights = {class_mapping[cn]['index']: w for cn, w in weights.items()}
    
    with open(metadata_dir / "class_weights.json", 'w') as f:
        json.dump(index_weights, f, indent=4)
        
    return index_weights, counts
""",

    "preprocessing/validate_preprocessing.py": r"""import os
import json
import hashlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
from preprocessing.preprocess import load_and_preprocess_image
from preprocessing.class_weights import calculate_class_weights

BASE_DIR = Path(r"C:\Users\moham\Music\OneDrive\Desktop\Major project new Tomato")
# UPDATED: Use 'dataset' instead of 'dataset_cleaned'
DATASET_DIR = BASE_DIR / "dataset"
AUDIT_DIR = BASE_DIR / "data_audit"
METADATA_DIR = BASE_DIR / "artifacts" / "model_metadata"

def discover_and_map_classes():
    """Dynamically discovers actual folder names and maps them to standard names."""
    train_dir = DATASET_DIR / "train"
    actual_folders = sorted([f.name for f in train_dir.iterdir() if f.is_dir()])
    
    expected_standards = [
        "Not_Tomato", "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Healthy",
        "Tomato_Late_blight", "Tomato_Leaf_Mold", "Tomato_Mosaic_Virus", 
        "Tomato_Septoria_Leaf_Spot", "Tomato_Spider_Mites", "Tomato_Target_Spot", 
        "Tomato_Yellow_Leaf_CurlVirus"
    ]
    
    class_mapping = {}
    matched_standards = set()
    
    # Map actual folders to standard names (case-insensitive matching)
    for actual in actual_folders:
        for std in expected_standards:
            if actual.lower() == std.lower():
                display = std.replace("_", " ").title()
                if std == "Not_Tomato": display = "Not Tomato"
                if std == "Tomato_Late_blight": display = "Late Blight"
                if std == "Tomato_Yellow_Leaf_CurlVirus": display = "Yellow Leaf Curl Virus"
                
                class_mapping[actual] = {
                    "index": len(class_mapping),
                    "standard_name": std,
                    "display_name": display
                }
                matched_standards.add(std)
                break
                
    return class_mapping, actual_folders

def setup_configs():
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    (AUDIT_DIR / "preprocessing_samples").mkdir(parents=True, exist_ok=True)
    
    class_mapping, actual_classes = discover_and_map_classes()
    
    if len(class_mapping) != 11:
        print(f"WARNING: Expected 11 classes, found {len(class_mapping)}.")
        
    with open(METADATA_DIR / "class_mapping.json", 'w', encoding='utf-8') as f:
        json.dump(class_mapping, f, indent=4)
        
    config = {
        "target_size": [224, 224],
        "batch_size": 32,
        "normalization": "min_max_0_1",
        "color_mode": "rgb"
    }
    with open(METADATA_DIR / "preprocessing_config.json", 'w') as f:
        json.dump(config, f, indent=4)
        
    return actual_classes

def run_validation():
    print("="*60)
    print(" PHASE 2B: DATASET PREPROCESSING VALIDATION (FIXED)")
    print("="*60)
    
    actual_classes = setup_configs()
    
    with open(METADATA_DIR / "class_mapping.json", 'r') as f:
        class_mapping = json.load(f)
        
    print(f"\n[1] Discovered {len(actual_classes)} actual classes.")
    for actual, info in class_mapping.items():
        print(f"    '{actual}' -> Index {info['index']} ('{info['display_name']}')")
    
    print("\n[2] Verifying class counts...")
    dist_data = []
    for split in ["train", "validation", "test"]:
        split_dir = DATASET_DIR / split
        for cn in actual_classes:
            cdir = split_dir / cn
            count = len(list(cdir.glob("*"))) if cdir.exists() else 0
            dist_data.append({"split": split, "class": cn, "count": count})
            
    df_dist = pd.DataFrame(dist_data)
    df_dist.to_csv(AUDIT_DIR / "final_class_distribution.csv", index=False)
    
    print("[3] Checking for cross-split duplicates (MD5)...")
    print("    NOTE: Documenting known dataset limitation (5,751 cross-split pairs).")
    file_hashes = {}
    cross_split_dups = 0
    for split in ["train", "validation", "test"]:
        split_dir = DATASET_DIR / split
        for cn in actual_classes:
            cdir = split_dir / cn
            if not cdir.exists(): continue
            for img_path in cdir.iterdir():
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    h = hashlib.md5(img_path.read_bytes()).hexdigest()
                    if h in file_hashes:
                        if file_hashes[h]["split"] != split:
                            cross_split_dups += 1
                    else:
                        file_hashes[h] = {"split": split, "class": cn, "file": img_path.name}
    print(f"   Cross-split duplicate pairs found: {cross_split_dups}")
    print("   STATUS: Documented as known limitation. Evaluation will note this leakage.")
    
    print("[4] Testing preprocessing pipeline...")
    # Find a valid image to test
    test_img_path = None
    for cn in actual_classes:
        cdir = DATASET_DIR / "train" / cn
        imgs = list(cdir.glob("*.jpg")) + list(cdir.glob("*.png"))
        if imgs:
            test_img_path = imgs[0]
            break
            
    if not test_img_path:
        raise Exception("No images found to test preprocessing pipeline!")
        
    img_array = load_and_preprocess_image(test_img_path)
    assert img_array is not None, "Failed to load image"
    assert img_array.shape == (224, 224, 3), f"Wrong shape: {img_array.shape}"
    assert img_array.dtype == np.float32, "Wrong dtype"
    assert img_array.min() >= 0.0 and img_array.max() <= 1.0, "Normalization failed"
    print("   Preprocessing pipeline test PASSED.")
    
    print("[5] Calculating class weights...")
    index_weights, counts = calculate_class_weights()
    print(f"   Weights saved to {METADATA_DIR / 'class_weights.json'}")
    
    print("[6] Generating visualizations...")
    for split in ["train", "validation", "test"]:
        split_df = df_dist[df_dist["split"] == split]
        plt.figure(figsize=(10, 6))
        # Use display names for better readability
        display_names = [class_mapping[cn]['display_name'] for cn in split_df['class']]
        plt.bar(display_names, split_df["count"], color='green')
        plt.xticks(rotation=45, ha='right')
        plt.title(f"{split.capitalize()} Class Distribution")
        plt.ylabel("Number of Images")
        plt.tight_layout()
        plt.savefig(AUDIT_DIR / f"{split}_distribution.png", dpi=150)
        plt.close()
        
    fig, axes = plt.subplots(11, 2, figsize=(10, 25))
    for i, cn in enumerate(actual_classes):
        cdir = DATASET_DIR / "train" / cn
        img_files = list(cdir.glob("*.jpg")) + list(cdir.glob("*.png"))
        if not img_files: 
            axes[i, 0].axis('off')
            axes[i, 1].axis('off')
            continue
        img_path = img_files[0]
        orig_img = Image.open(img_path)
        proc_img = load_and_preprocess_image(img_path)
        
        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_title(f"Original: {class_mapping[cn]['display_name']}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(proc_img)
        axes[i, 1].set_title(f"Preprocessed (224x224)")
        axes[i, 1].axis('off')
        
    plt.tight_layout()
    plt.savefig(AUDIT_DIR / "preprocessing_samples" / "preprocessing_samples_grid.png", dpi=150)
    plt.close()
    
    report = {
        "status": "SUCCESS",
        "classes_verified": len(actual_classes) == 11,
        "actual_classes": actual_classes,
        "cross_split_duplicates_documented": cross_split_dups,
        "preprocessing_test_passed": True,
        "target_size": [224, 224],
        "batch_size": 32,
        "class_weights_calculated": True,
        "dataset_source": "dataset/"
    }
    with open(AUDIT_DIR / "preprocessing_report.json", 'w') as f:
        json.dump(report, f, indent=4)
        
    print("\n" + "="*60)
    print(" PHASE 2B VALIDATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    run_validation()
"""
}

print("Updating preprocessing modules...")
for rel_path, content in files_content.items():
    file_path = BASE_DIR / rel_path
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Updated: {rel_path}")
print("Update complete.")