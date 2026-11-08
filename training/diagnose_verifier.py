import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path

from preprocessing.dataset_loader import collect_image_paths
from preprocessing.preprocess import (
    IMG_SIZE,
    BATCH_SIZE,
    CLASS_NAMES
)

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    ROOT / "artifacts" / "phase3" / "verifier" /
    "run_20261107_170126_917465" / "best_verifier.keras"
)


def decode_and_normalize(path, label):
    image = tf.io.read_file(path)

    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    image = tf.clip_by_value(image, 0.0, 255.0)
    image = image / 255.0

    return image, tf.cast(label, tf.int32)


def main():
    print("=" * 55)
    print("TomatoAI - Unaugmented Training Diagnostic")
    print("=" * 55)

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)

    print("\nLoading existing best model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("\nCollecting original training images...")
    paths, labels = collect_image_paths("train")

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    dataset = dataset.map(
        decode_and_normalize,
        num_parallel_calls=1,
        deterministic=True
    )

    dataset = dataset.batch(BATCH_SIZE).prefetch(1)

    print("\nEvaluating unaugmented training images...")
    probabilities = model.predict(dataset, verbose=1)

    y_true = np.asarray(labels, dtype=np.int32)
    y_pred = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(y_true, y_pred)

    print("\nUNaugmented TRAINING ACCURACY:")
    print(f"{accuracy * 100:.2f}%")

    print("\nCLASSIFICATION REPORT:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    print("\nDiagnostic completed.")
    print("No model or original dataset was modified.")


if __name__ == "__main__":
    main()
