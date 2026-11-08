import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from preprocessing.dataset_loader import (
    collect_image_paths,
    decode_and_resize
)
from preprocessing.preprocess import IMG_SIZE, CLASS_NAMES


ROOT = Path(__file__).resolve().parent.parent

V2_PATH = (
    ROOT / "artifacts" / "phase3" / "verifier_v2"
    / "run_20261107_190658_190660"
    / "best_verifier_v2.keras"
)

BATCH_SIZE = 16
PCA_COMPONENTS = 256
SEED = 42


def create_run_dir():
    parent = ROOT / "artifacts" / "phase3" / "classical_ml"
    parent.mkdir(parents=True, exist_ok=True)

    run = parent / ("run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    run.mkdir()
    return run


def make_dataset(split):
    paths, labels = collect_image_paths(split)

    ds = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    ds = ds.map(
        decode_and_resize,
        num_parallel_calls=1,
        deterministic=True
    )

    ds = ds.batch(BATCH_SIZE)


    return ds, np.asarray(labels, dtype=np.int32)


def extract_features(feature_model, split, output_dir):
    cache_x = output_dir / f"{split}_features.npy"
    cache_y = output_dir / f"{split}_labels.npy"

    if cache_x.exists() and cache_y.exists():
        print(f"Loading cached {split} features...")
        return np.load(cache_x), np.load(cache_y)

    ds, labels = make_dataset(split)

    features = []

    print(f"\nExtracting {split} features...")

    for images, _ in ds:
        batch_features = feature_model(
            images,
            training=False
        ).numpy()

        features.append(batch_features)

    X = np.concatenate(features).astype(np.float32)

    if len(X) != len(labels):
        raise ValueError(f"{split}: feature/label count mismatch")

    np.save(cache_x, X)
    np.save(cache_y, labels)

    print(f"{split}: {X.shape}")

    return X, labels


def build_models():
    return {
        "SVM": CalibratedClassifierCV(
            estimator=LinearSVC(
                C=1.0,
                max_iter=5000,
                random_state=SEED
            ),
            method="sigmoid",
            cv=3
        ),

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=SEED
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=120,
            max_features="sqrt",
            min_samples_leaf=2,
            n_jobs=2,
            random_state=SEED
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            n_jobs=2
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=30,
            min_samples_leaf=2,
            random_state=SEED
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=160,
            max_features="sqrt",
            min_samples_leaf=2,
            n_jobs=2,
            random_state=SEED
        ),

        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=80,
            max_leaf_nodes=31,
            learning_rate=0.1,
            random_state=SEED
        )
    }


def main():
    print("=" * 65)
    print("TomatoAI - Seven Classical ML Models")
    print("=" * 65)

    if not V2_PATH.is_file():
        raise FileNotFoundError(V2_PATH)

    output_dir = create_run_dir()

    print("\nLoading trained MobileNetV2 V2...")
    base_model = tf.keras.models.load_model(V2_PATH)

    feature_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=base_model.get_layer(
            "global_average_pooling2d"
        ).output
    )

    print("\nFeature extractor output:", feature_model.output_shape)

    X_train, y_train = extract_features(
        feature_model, "train", output_dir
    )

    X_val, y_val = extract_features(
        feature_model, "validation", output_dir
    )

    print("\nFitting PCA on training features only...")

    pca = PCA(
        n_components=PCA_COMPONENTS,
        svd_solver="randomized",
        random_state=SEED
    )

    X_train_pca = pca.fit_transform(X_train).astype(np.float32)
    X_val_pca = pca.transform(X_val).astype(np.float32)

    joblib.dump(
        pca,
        output_dir / "pca.joblib",
        compress=3
    )

    print("Training shape:", X_train_pca.shape)
    print("Validation shape:", X_val_pca.shape)

    del X_train, X_val

    models = build_models()
    results = []

    for name, classifier in models.items():
        print("\n" + "=" * 55)
        print("TRAINING:", name)
        print("=" * 55)

        start = time.perf_counter()

        classifier.fit(X_train_pca, y_train)

        training_seconds = time.perf_counter() - start

        start = time.perf_counter()
        predictions = classifier.predict(X_val_pca)
        inference_seconds = (
            time.perf_counter() - start
        ) / len(y_val) * 1000

        accuracy = accuracy_score(y_val, predictions)
        macro_precision = precision_score(
            y_val, predictions, average="macro", zero_division=0
        )
        macro_recall = recall_score(
            y_val, predictions, average="macro", zero_division=0
        )
        macro_f1 = f1_score(
            y_val, predictions, average="macro", zero_division=0
        )
        weighted_f1 = f1_score(
            y_val, predictions, average="weighted", zero_division=0
        )

        model_filename = (
            name.lower().replace(" ", "_") + ".joblib"
        )

        joblib.dump(
            classifier,
            output_dir / model_filename,
            compress=3
        )

        result = {
            "model": name,
            "accuracy": float(accuracy),
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "training_seconds": round(training_seconds, 3),
            "inference_ms_per_image": round(inference_seconds, 4),
            "model_file": model_filename
        }

        results.append(result)

        print(json.dumps(result, indent=2))

        del predictions

    results.sort(
        key=lambda item: item["macro_f1"],
        reverse=True
    )

    summary = {
        "project": "TomatoAI",
        "feature_extractor": "MobileNetV2 V2",
        "feature_dimensions": 1280,
        "pca_dimensions": PCA_COMPONENTS,
        "training_images": int(len(y_train)),
        "validation_images": int(len(y_val)),
        "class_names": CLASS_NAMES,
        "selection_metric": "validation_macro_f1",
        "models": results,
        "warning": (
            "Validation results are used for model selection. "
            "Cross-split duplicate images were identified in the dataset. "
            "Test results must be interpreted with this limitation."
        )
    }

    with (output_dir / "comparison_results.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=4)

    with (output_dir / "comparison_results.txt").open(
        "w", encoding="utf-8"
    ) as file:
        for index, result in enumerate(results, start=1):
            file.write(
                f"{index}. {result['model']}: "
                f"Accuracy={result['accuracy']:.4f}, "
                f"Macro F1={result['macro_f1']:.4f}, "
                f"Weighted F1={result['weighted_f1']:.4f}\n"
            )

    print("\n" + "=" * 65)
    print("CLASSICAL MODEL TRAINING COMPLETED")
    print("=" * 65)
    print("Results:", output_dir)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

