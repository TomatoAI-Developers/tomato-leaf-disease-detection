import json
from pathlib import Path
from datetime import datetime

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from preprocessing.dataset_loader import load_datasets
from preprocessing.preprocess import CLASS_NAMES


ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    ROOT / "artifacts" / "phase3" / "verifier_v2" /
    "run_20261107_190658_190660" / "best_verifier_v2.keras"
)


def create_output_dir():
    parent = ROOT / "artifacts" / "phase3" / "verifier_v2_evaluation"
    parent.mkdir(parents=True, exist_ok=True)

    while True:
        path = parent / (
            "evaluation_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue


def main():
    print("=" * 55)
    print("TomatoAI - Verifier V2 Test Evaluation")
    print("=" * 55)

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    print("\nLoading saved best verifier...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("\nLoading test dataset...")
    _, _, test_ds, class_names = load_datasets()

    if class_names != CLASS_NAMES:
        raise ValueError("Class mapping mismatch.")

    y_true = np.concatenate([
        labels.numpy() for _, labels in test_ds
    ])

    print("\nPredicting test images...")
    probabilities = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)

    if len(y_true) != len(y_pred):
        raise ValueError("Prediction count does not match test labels.")

    accuracy = float(accuracy_score(y_true, y_pred))

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES)))
    )

    output_dir = create_output_dir()

    # Save detailed classification report.
    with (output_dir / "classification_report.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=4)

    with (output_dir / "classification_report.txt").open(
        "w", encoding="utf-8"
    ) as f:
        f.write(report_text)

    # Save confusion matrix data.
    np.savetxt(
        output_dir / "confusion_matrix.csv",
        matrix,
        delimiter=",",
        fmt="%d"
    )

    # Save confusion matrix visualization.
    fig, ax = plt.subplots(figsize=(13, 11))
    image = ax.imshow(matrix, interpolation="nearest")
    fig.colorbar(image, ax=ax)

    ax.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted class",
        ylabel="True class",
        title="Tomato Verifier - Test Confusion Matrix"
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=70,
        ha="right",
        rotation_mode="anchor"
    )

    threshold = matrix.max() / 2.0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j, i, str(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
                fontsize=8
            )

    fig.tight_layout()
    fig.savefig(
        output_dir / "confusion_matrix.png",
        dpi=160,
        bbox_inches="tight"
    )
    plt.close(fig)

    # Save summary.
    summary = {
        "model": "MobileNetV2 Tomato Verifier V2",
        "test_images": int(len(y_true)),
        "class_count": len(CLASS_NAMES),
        "test_accuracy": accuracy,
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "evaluation_warning": (
            "Dataset contains cross-split exact duplicates. "
            "Test results are not independent or leakage-free."
        )
    }

    with (output_dir / "evaluation_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=4)

    print("\nCLASSIFICATION REPORT")
    print(report_text)

    print("\nEVALUATION SUMMARY")
    print(json.dumps(summary, indent=4))

    print("\nResults saved to:", output_dir)


if __name__ == "__main__":
    main()


