import json
from pathlib import Path
from .preprocess import CLASS_NAMES, DISEASE_CLASS_NAMES, SUPPORTED_EXTENSIONS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"

def count_images(directory):
    """Counts only supported image files in a directory."""
    if not directory.exists():
        return 0
    return sum(1 for p in directory.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)

def calculate_class_weights(class_list, split_name="train"):
    """Calculates inverse frequency weights for a given list of classes."""
    train_dir = DATASET_ROOT / split_name
    counts = {cls: count_images(train_dir / cls) for cls in class_list}
    
    # Strict check: raise error if any class has 0 images
    for cls, count in counts.items():
        if count == 0:
            raise ValueError(f"Class '{cls}' has 0 supported images in {split_name} split. Cannot calculate weights.")
    
    total_samples = sum(counts.values())
    num_classes = len(class_list)
    
    weights = {}
    for i, cls in enumerate(class_list):
        weights[i] = float(total_samples / (num_classes * counts[cls]))
    return weights

def get_all_weights(output_dir=None):
    """Calculates and saves weights for both 11-class and 10-class models."""
    if output_dir is None:
        output_dir = PROJECT_ROOT / "artifacts" / "phase2b"
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 11-class weights for Tomato Verifier
    verifier_weights = calculate_class_weights(CLASS_NAMES)
    with open(output_dir / "verifier_class_weights.json", "w") as f:
        json.dump(verifier_weights, f, indent=4)
        
    # 10-class weights for Disease Classifier
    disease_weights = calculate_class_weights(DISEASE_CLASS_NAMES)
    with open(output_dir / "disease_class_weights.json", "w") as f:
        json.dump(disease_weights, f, indent=4)
        
    return verifier_weights, disease_weights