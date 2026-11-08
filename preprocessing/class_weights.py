"""Class weights with exclusive, no-overwrite JSON output."""
import json
from pathlib import Path
from .preprocess import CLASS_NAMES, DISEASE_CLASS_NAMES, SUPPORTED_EXTENSIONS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"

def count_images(directory):
    directory = Path(directory)
    if not directory.is_dir():
        return 0
    return sum(1 for p in directory.iterdir()
               if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)

def calculate_class_weights(class_list, split_name="train"):
    split_dir = DATASET_ROOT / split_name
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing dataset split: {split_dir}")
    counts = {name: count_images(split_dir / name) for name in class_list}
    missing = [name for name, count in counts.items() if count == 0]
    if missing:
        raise ValueError(f"Zero supported images for classes: {missing}")
    total = sum(counts.values())
    n_classes = len(class_list)
    return {i: float(total / (n_classes * counts[name]))
            for i, name in enumerate(class_list)}

def _write_pair_exclusive(directory, verifier, disease):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payloads = [
        (directory / "verifier_class_weights.json", verifier),
        (directory / "disease_class_weights.json", disease),
    ]
    existing = [str(path) for path, _ in payloads if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite: " + ", ".join(existing))
    created = []
    try:
        for path, payload in payloads:
            with path.open("x", encoding="utf-8") as handle:
                created.append(path)
                json.dump(payload, handle, indent=4)
                handle.write("\n")
    except Exception:
        for path in created:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise

def get_all_weights(output_dir=None):
    """No argument preserves historical default path; all writes are exclusive."""
    verifier = calculate_class_weights(CLASS_NAMES)
    disease = calculate_class_weights(DISEASE_CLASS_NAMES)
    destination = (Path(output_dir) if output_dir is not None
                   else PROJECT_ROOT / "artifacts" / "phase2b")
    _write_pair_exclusive(destination, verifier, disease)
    return verifier, disease
