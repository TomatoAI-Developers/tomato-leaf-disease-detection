import tensorflow as tf
from pathlib import Path

from preprocessing.dataset_loader import load_datasets

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    ROOT / "artifacts" / "phase3" / "verifier" /
    "run_20261107_170126_917465" / "best_verifier.keras"
)

def main():
    print("=" * 55)
    print("TomatoAI - Augmentation Impact Diagnostic")
    print("=" * 55)

    model = tf.keras.models.load_model(MODEL_PATH)

    print("\nLoading datasets...")
    train_ds, _, _, _ = load_datasets()

    print("\nEvaluating model on augmented training images...")
    results = model.evaluate(train_ds, verbose=1)

    for name, value in zip(model.metrics_names, results):
        print(f"{name}: {value}")

    print("\nDiagnostic completed. No training performed.")
    print("=" * 55)

if __name__ == "__main__":
    main()
