import os
import time
import joblib
import numpy as np
import tensorflow as tf

from pathlib import Path
from functools import lru_cache
from tensorflow.keras.models import load_model
from preprocessing.dataset_loader import decode_and_resize


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLASSICAL_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "phase3"
    / "classical_ml"
    / "run_20261107_211856"
)

MOBILENET_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "phase3"
    / "verifier_v2"
    / "run_20261107_190658_190660"
    / "best_verifier_v2.keras"
)

CLASS_NAMES = [
    "Not_Tomato",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Mosaic_Virus",
    "Tomato_Septoria_Leaf_Spot",
    "Tomato_Spider_Mites",
    "Tomato_Target_Spot",
    "Tomato_Yellow_Leaf_CurlVirus",
]

CLASSIFIER_FILES = {
    "SVM": "svm.joblib",
    "KNN": "knn.joblib",
    "Logistic Regression": "logistic_regression.joblib",
    "Random Forest": "random_forest.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Extra Trees": "extra_trees.joblib",
    "HistGradientBoosting": "gradient_boosting.joblib",
}


@lru_cache(maxsize=1)
def load_prediction_assets():
    """Load all trained models and preprocessing assets once."""

    required_files = [
        MOBILENET_PATH,
        CLASSICAL_DIR / "pca.joblib",
    ]

    required_files.extend(
        CLASSICAL_DIR / filename
        for filename in CLASSIFIER_FILES.values()
    )

    missing = [str(path) for path in required_files if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "Required model files are missing:\n" + "\n".join(missing)
        )

    print("Loading MobileNetV2 V2...")
    verifier = load_model(MOBILENET_PATH, compile=False)

    feature_layer = verifier.get_layer("global_average_pooling2d")
    feature_extractor = tf.keras.Model(
        inputs=verifier.input,
        outputs=feature_layer.output,
        name="mobilenetv2_feature_extractor",
    )

    print("Loading PCA...")
    pca = joblib.load(CLASSICAL_DIR / "pca.joblib")

    classifiers = {}

    for model_name, filename in CLASSIFIER_FILES.items():
        print(f"Loading {model_name}...")
        classifiers[model_name] = joblib.load(
            CLASSICAL_DIR / filename
        )

    return verifier, feature_extractor, pca, classifiers


def prepare_image(image_path):
    """Use the project's existing image decoder and normalization."""

    image_path = str(Path(image_path).resolve())

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image, _ = decode_and_resize(
        tf.constant(image_path),
        tf.constant(0, dtype=tf.int32),
    )

    image = tf.expand_dims(image, axis=0)
    return image


def top_predictions(probabilities, limit=3):
    probabilities = np.asarray(probabilities).reshape(-1)

    indices = np.argsort(probabilities)[::-1][:limit]

    return [
        {
            "class": CLASS_NAMES[int(index)],
            "confidence": round(float(probabilities[index]) * 100, 2),
        }
        for index in indices
    ]


def predict_image(image_path):
    """
    Run MobileNetV2 V2 and all seven classical classifiers.

    Confidence is each model's own predicted-class probability.
    It is not the same as the model's test accuracy.
    """

    verifier, feature_extractor, pca, classifiers = (
        load_prediction_assets()
    )

    image = prepare_image(image_path)

    results = {}

    # MobileNetV2 V2 prediction
    start = time.perf_counter()

    mobilenet_probabilities = verifier.predict(
        image,
        verbose=0,
    )[0]

    mobilenet_elapsed = (
        time.perf_counter() - start
    ) * 1000

    mobilenet_index = int(np.argmax(mobilenet_probabilities))

    results["MobileNetV2 V2"] = {
        "predicted_class": CLASS_NAMES[mobilenet_index],
        "confidence": round(
            float(mobilenet_probabilities[mobilenet_index]) * 100,
            2,
        ),
        "top3": top_predictions(mobilenet_probabilities),
        "inference_ms": round(mobilenet_elapsed, 3),
    }

    # Shared MobileNetV2 feature extraction for classical models
    feature_start = time.perf_counter()

    features = feature_extractor.predict(
        image,
        verbose=0,
    )

    feature_elapsed = (
        time.perf_counter() - feature_start
    ) * 1000

    reduced_features = pca.transform(features)

    # Seven actual trained classifiers
    for model_name, classifier in classifiers.items():

        start = time.perf_counter()

        predicted_index = int(
            classifier.predict(reduced_features)[0]
        )

        probabilities = classifier.predict_proba(
            reduced_features
        )[0]

        elapsed = (time.perf_counter() - start) * 1000

        # Locate the probability belonging to the predicted class.
        class_positions = np.where(
            classifier.classes_ == predicted_index
        )[0]

        if len(class_positions) == 0:
            raise ValueError(
                f"Predicted class missing from {model_name} classes_"
            )

        confidence = float(
            probabilities[class_positions[0]]
        )

        # Convert the classifier's integer label to the class name.
        if predicted_index < 0 or predicted_index >= len(CLASS_NAMES):
            raise ValueError(
                f"Invalid class index from {model_name}: "
                f"{predicted_index}"
            )

        results[model_name] = {
            "predicted_class": CLASS_NAMES[predicted_index],
            "confidence": round(confidence * 100, 2),
            "top3": top_predictions(probabilities),
            "classifier_inference_ms": round(elapsed, 3),
            "shared_feature_extraction_ms": round(
                feature_elapsed, 3
            ),
        }

    return {
        "image": os.path.basename(str(image_path)),
        "class_names": CLASS_NAMES,
        "models": results,
        "model_count": len(results),
        "feature_extractor": "MobileNetV2 V2",
        "pca_dimensions": int(pca.n_components_),
        "note": (
            "Confidence is a model probability, not test accuracy. "
            "Classical models share one feature-extraction pass."
        ),
    }


if __name__ == "__main__":
    print("Prediction engine module loaded.")
    print("Available models:")
    for name in ["MobileNetV2 V2"] + list(CLASSIFIER_FILES.keys()):
        print(" -", name)

    print("\nRun an image test with:")
    print(
        'python -c "from inference.predictor import predict_image; '
        'import json; print(json.dumps(predict_image(r'
        "'YOUR_IMAGE_PATH'"
        '), indent=2))"'
    )
