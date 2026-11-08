"""Read-only dataset validation; outputs only to a new Phase 2B run folder."""
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from .class_weights import get_all_weights
from .dataset_loader import DATASET_ROOT, load_datasets
from .preprocess import (
    BATCH_SIZE, CLASS_NAMES, DISEASE_CLASS_NAMES, IMG_SIZE,
    SUPPORTED_EXTENSIONS, load_and_preprocess_image,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "phase2b"
SPLITS = ("train", "validation", "test")

def create_run_dir():
    """Create a unique run directory atomically; never reuse an existing one."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    counter = 0
    while True:
        name = f"run_{stamp}" if counter == 0 else f"run_{stamp}_{counter}"
        candidate = ARTIFACTS_DIR / name
        try:
            candidate.mkdir(parents=False, exist_ok=False)
            return candidate
        except FileExistsError:
            counter += 1

def list_images(directory):
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir()
                  if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)

def validate_dataset_structure():
    counts = {}
    for split in SPLITS:
        split_dir = DATASET_ROOT / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split: {split_dir}")
        folders = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
        if folders != CLASS_NAMES:
            raise ValueError(f"{split} folders mismatch. Expected {CLASS_NAMES}; got {folders}")
        counts[split] = {}
        for name in CLASS_NAMES:
            count = len(list_images(split_dir / name))
            if count == 0:
                raise ValueError(f"No supported images in {split}/{name}")
            counts[split][name] = count
    print("[PASS] Dataset folders and per-class image counts.")
    return counts

def validate_representative_images():
    checked = 0
    for split in SPLITS:
        for name in CLASS_NAMES:
            paths = list_images(DATASET_ROOT / split / name)
            if not paths:
                raise ValueError(f"No image in {split}/{name}")
            arr = load_and_preprocess_image(paths[0])
            assert arr.shape == (224, 224, 3)
            assert arr.dtype == np.float32
            assert np.isfinite(arr).all() and 0.0 <= arr.min() <= arr.max() <= 1.0
            checked += 1
    print(f"[PASS] {checked} representative images decoded and normalized.")

def assert_batch(images, labels, split):
    if images.shape.rank != 4 or tuple(images.shape[1:]) != (224, 224, 3):
        raise AssertionError(f"{split}: invalid batch shape {images.shape}")
    if images.dtype != tf.float32:
        raise AssertionError(f"{split}: images must be float32")
    lo, hi = float(tf.reduce_min(images).numpy()), float(tf.reduce_max(images).numpy())
    if lo < 0.0 or hi > 1.0:
        raise AssertionError(f"{split}: pixel range [{lo},{hi}] outside [0,1]")
    min_label, max_label = int(tf.reduce_min(labels).numpy()), int(tf.reduce_max(labels).numpy())
    if min_label < 0 or max_label >= len(CLASS_NAMES):
        raise AssertionError(f"{split}: invalid label range {min_label}..{max_label}")

def validate_data_pipeline(train_ds, val_ds, test_ds):
    seen, batches, images_checked = set(), 0, 0
    for images, labels in train_ds:
        assert_batch(images, labels, "train")
        seen.update(int(v) for v in labels.numpy())
        batches += 1
        images_checked += int(labels.shape[0])
        if seen == set(range(len(CLASS_NAMES))):
            break
    if seen != set(range(len(CLASS_NAMES))):
        raise RuntimeError(f"Training smoke test missed classes; labels seen: {sorted(seen)}")
    print(f"[PASS] Train smoke test: {batches} batches / {images_checked} images; all 11 labels seen.")

    for ds, split in ((val_ds, "validation"), (test_ds, "test")):
        first, second = next(iter(ds), None), next(iter(ds), None)
        if first is None or second is None:
            raise RuntimeError(f"{split} dataset is empty")
        x1, y1 = first
        x2, y2 = second
        assert_batch(x1, y1, split)
        if not np.array_equal(x1[0].numpy(), x2[0].numpy()):
            raise AssertionError(f"{split} is not deterministic; inspect shuffle/augmentation")
        if not np.array_equal(y1.numpy(), y2.numpy()):
            raise AssertionError(f"{split} labels are not deterministic")
        print(f"[PASS] {split}: deterministic, unaugmented pipeline smoke test.")

def md5_file(path, chunk_size=1024 * 1024):
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()

def verify_cross_split_duplicates():
    grouped = {}
    for split in SPLITS:
        for folder in (DATASET_ROOT / split).iterdir():
            if not folder.is_dir():
                continue
            for path in list_images(folder):
                digest = md5_file(path)
                grouped.setdefault(digest, Counter())[split] += 1

    pair_count = 0
    multi_split_hashes = 0
    for counts in grouped.values():
        train, val, test = (counts.get(s, 0) for s in SPLITS)
        pair_count += train * val + train * test + val * test
        if sum(value > 0 for value in (train, val, test)) > 1:
            multi_split_hashes += 1
    return {
        "unique_md5_hashes": len(grouped),
        "cross_split_exact_duplicate_pairs": pair_count,
        "hashes_present_in_multiple_splits": multi_split_hashes,
        "method": "Chunked MD5; cross-split pair combinations only",
    }

def write_json_exclusive(path, data):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4)
        handle.write("\n")

def generate_reports(run_dir, counts, duplicate_report):
    run_dir = Path(run_dir)
    with (run_dir / "class_distribution.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Class", "Train", "Validation", "Test"])
        for name in CLASS_NAMES:
            writer.writerow([name, counts["train"][name],
                             counts["validation"][name], counts["test"][name]])

    mapping_11 = {name: i for i, name in enumerate(CLASS_NAMES)}
    mapping_10 = {name: i for i, name in enumerate(DISEASE_CLASS_NAMES)}
    assert len(mapping_11) == 11 and list(mapping_11.values()) == list(range(11))
    assert len(mapping_10) == 10 and list(mapping_10.values()) == list(range(10))
    assert "Not_Tomato" not in mapping_10
    write_json_exclusive(run_dir / "class_mapping_11.json", mapping_11)
    write_json_exclusive(run_dir / "disease_class_mapping_10.json", mapping_10)

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "image_size": list(IMG_SIZE),
        "batch_size": BATCH_SIZE,
        "verifier_class_count": 11,
        "disease_class_count": 10,
        "verifier_class_names": CLASS_NAMES,
        "disease_class_names": DISEASE_CLASS_NAMES,
        "counts_by_split": counts,
        "augmentation": "training split only",
        "normalization": "float32 [0,1], clipped after augmentation",
        "duplicate_scan": duplicate_report,
        "evaluation_warning": (
            "Phase 1 found cross-split duplicates. Do not describe test results "
            "as independent or leakage-free."
        ),
        "preservation": (
            "Original dataset, data_audit, and artifacts/model_metadata are not modified."
        ),
    }
    write_json_exclusive(run_dir / "preprocessing_report.json", report)

def save_visual_samples(train_ds, run_dir, maximum=3):
    samples_dir = Path(run_dir) / "preprocessing_samples"
    samples_dir.mkdir(exist_ok=False)
    images, labels = next(iter(train_ds))
    count = min(maximum, int(images.shape[0]))
    for i in range(count):
        array = np.clip(images[i].numpy(), 0.0, 1.0)
        label = int(labels[i].numpy())
        name = CLASS_NAMES[label]
        output = samples_dir / f"train_sample_{i}_{name}.png"
        plt.figure(figsize=(4, 4))
        plt.imshow(array)
        plt.title(f"Training sample: {name}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output, dpi=120)
        plt.close()
    print(f"[PASS] Saved {count} training samples.")

def main():
    print("=== TomatoAI Phase 2B validation ===")
    # Read-only checks first; no output folder is created if these fail.
    counts = validate_dataset_structure()
    validate_representative_images()
    train_ds, val_ds, test_ds, names = load_datasets()
    if names != CLASS_NAMES:
        raise AssertionError("Unexpected class names returned by dataset loader")
    validate_data_pipeline(train_ds, val_ds, test_ds)

    print("[INFO] Running exact duplicate scan; original files are read-only.")
    duplicates = verify_cross_split_duplicates()

    # Create a fresh output directory only after read-only checks pass.
    run_dir = create_run_dir()
    print(f"[INFO] Outputs will be written only to: {run_dir}")

    get_all_weights(run_dir)
    generate_reports(run_dir, counts, duplicates)
    save_visual_samples(train_ds, run_dir)

    print(f"=== Phase 2B completed. Results: {run_dir} ===")
    print("No original dataset, Phase 1 audit, or existing model metadata was modified.")

if __name__ == "__main__":
    main()
