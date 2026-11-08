import json
import time
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from preprocessing.dataset_loader import collect_image_paths, decode_and_resize
from preprocessing.preprocess import CLASS_NAMES

ROOT = Path(__file__).resolve().parent.parent

RUN = ROOT / "artifacts" / "phase3" / "classical_ml" / "run_20261107_211856"
V2 = ROOT / "artifacts" / "phase3" / "verifier_v2" / "run_20261107_190658_190660" / "best_verifier_v2.keras"

BATCH_SIZE = 16

def main():
    print("Loading PCA and trained classifiers...")
    pca = joblib.load(RUN / "pca.joblib")

    base = tf.keras.models.load_model(V2)
    extractor = tf.keras.Model(
        base.input,
        base.get_layer("global_average_pooling2d").output
    )

    paths, y_true = collect_image_paths("test")
    y_true = np.asarray(y_true, dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices(paths)
    def decode(path):
        image, _ = decode_and_resize(path, tf.constant(0))
        return image

    ds = ds.map(decode, num_parallel_calls=1)
    ds = ds.batch(BATCH_SIZE)

    print("Extracting test features...")
    batches = []
    for images in ds:
        batches.append(extractor(images, training=False).numpy())

    X = np.concatenate(batches).astype(np.float32)

    if len(X) != len(y_true):
        raise ValueError("Test feature and label counts differ.")

    X_test = pca.transform(X).astype(np.float32)
    del X, batches

    output = RUN / "test_evaluation"
    output.mkdir(exist_ok=True)

    model_files = {
        "SVM": "svm.joblib",
        "Logistic Regression": "logistic_regression.joblib",
        "Random Forest": "random_forest.joblib",
        "KNN": "knn.joblib",
        "Decision Tree": "decision_tree.joblib",
        "Extra Trees": "extra_trees.joblib",
        "Gradient Boosting": "gradient_boosting.joblib"
    }

    results = []

    for name, filename in model_files.items():
        print("\nEvaluating:", name)
        model = joblib.load(RUN / filename)

        start = time.perf_counter()
        predictions = model.predict(X_test)
        elapsed = time.perf_counter() - start

        accuracy = accuracy_score(y_true, predictions)
        report = classification_report(
            y_true,
            predictions,
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0
        )

        matrix = confusion_matrix(
            y_true,
            predictions,
            labels=list(range(len(CLASS_NAMES)))
        )

        safe_name = filename.replace(".joblib", "")

        with (output / f"{safe_name}_report.json").open("w") as f:
            json.dump(report, f, indent=4)

        np.savetxt(
            output / f"{safe_name}_confusion.csv",
            matrix,
            delimiter=",",
            fmt="%d"
        )

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(matrix, interpolation="nearest")
        ax.set(
            xticks=np.arange(len(CLASS_NAMES)),
            yticks=np.arange(len(CLASS_NAMES)),
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            xlabel="Predicted",
            ylabel="Actual",
            title=f"{name} - Test Confusion Matrix"
        )
        plt.setp(ax.get_xticklabels(), rotation=75, ha="right")
        fig.tight_layout()
        fig.savefig(output / f"{safe_name}_confusion.png", dpi=130)
        plt.close(fig)

        result = {
            "model": name,
            "test_accuracy": float(accuracy),
            "macro_precision": float(report["macro avg"]["precision"]),
            "macro_recall": float(report["macro avg"]["recall"]),
            "macro_f1": float(report["macro avg"]["f1-score"]),
            "weighted_f1": float(report["weighted avg"]["f1-score"]),
            "classifier_inference_ms_per_image": float(
                elapsed / len(y_true) * 1000
            )
        }
        results.append(result)
        print(json.dumps(result, indent=2))

    results.sort(key=lambda x: x["macro_f1"], reverse=True)

    summary = {
        "evaluation": "test",
        "test_images": int(len(y_true)),
        "class_count": len(CLASS_NAMES),
        "feature_extractor": "MobileNetV2 V2",
        "pca_dimensions": 256,
        "results": results,
        "warning": (
            "Cross-split exact duplicates were identified. "
            "These test scores are not independent or leakage-free."
        )
    }

    with (output / "test_comparison.json").open("w") as f:
        json.dump(summary, f, indent=4)

    print("\nFINAL CLASSICAL TEST LEADERBOARD")
    for index, item in enumerate(results, 1):
        print(
            f"{index}. {item['model']}: "
            f"Accuracy={item['test_accuracy']:.4%}, "
            f"Macro F1={item['macro_f1']:.4%}"
        )

    print("\nSaved:", output)

if __name__ == "__main__":
    main()
