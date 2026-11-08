import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import csv
from pathlib import Path
from datetime import datetime

import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocessing.dataset_loader import load_datasets
from preprocessing.preprocess import CLASS_NAMES

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_FILE = (
    ROOT / "artifacts" / "phase2b" /
    "run_20261107_164839_616105" /
    "verifier_class_weights.json"
)

EPOCHS = 15
LEARNING_RATE = 0.0001
SEED = 42

tf.keras.utils.set_random_seed(SEED)

# Keep CPU thread usage controlled.
try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except RuntimeError:
    pass

def create_run_directory():
    parent = ROOT / "artifacts" / "phase3" / "verifier"
    parent.mkdir(parents=True, exist_ok=True)

    while True:
        name = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = parent / name
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            continue


def load_class_weights():
    if not WEIGHTS_FILE.is_file():
        raise FileNotFoundError(
            f"Class weights not found: {WEIGHTS_FILE}"
        )

    with WEIGHTS_FILE.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    weights = {int(k): float(v) for k, v in raw.items()}

    expected = set(range(len(CLASS_NAMES)))
    if set(weights) != expected:
        raise ValueError("Class-weight indices do not match class mapping.")

    return weights


def build_model():
    inputs = tf.keras.Input(shape=(224, 224, 3), name="image")

    # Convert [0,1] inputs to MobileNetV2's expected [-1,1].
    x = tf.keras.layers.Rescaling(
        scale=2.0,
        offset=-1.0,
        name="mobilenet_normalization"
    )(inputs)

    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

    backbone.trainable = False

    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    outputs = tf.keras.layers.Dense(
        len(CLASS_NAMES),
        activation="softmax",
        dtype="float32",
        name="verifier_predictions"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="TomatoAI_Verifier_MobileNetV2"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def save_history(history, run_dir):
    history_data = history.history

    csv_path = run_dir / "training_history.csv"

    keys = list(history_data.keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + keys)

        for i in range(len(history_data["loss"])):
            writer.writerow(
                [i + 1] + [history_data[k][i] for k in keys]
            )

    plt.figure()
    plt.plot(history_data["accuracy"], label="Training")
    plt.plot(history_data["val_accuracy"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Tomato Verifier Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(run_dir / "accuracy_curve.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(history_data["loss"], label="Training")
    plt.plot(history_data["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Tomato Verifier Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(run_dir / "loss_curve.png", dpi=150)
    plt.close()


def main():
    print("=" * 55)
    print("TomatoAI - 11 Class Tomato Verifier Training")
    print("=" * 55)

    if len(CLASS_NAMES) != 11:
        raise ValueError("Expected exactly 11 verifier classes.")

    class_weights = load_class_weights()

    print("\nLoading datasets...")
    train_ds, val_ds, test_ds, class_names = load_datasets()

    print("\nClasses:", len(class_names))
    print("Training batches:", len(train_ds))
    print("Validation batches:", len(val_ds))
    print("Test batches:", len(test_ds))

    print("\nBuilding MobileNetV2 model...")
    model = build_model()
    model.summary()

    run_dir = create_run_directory()
    print("\nOutput directory:", run_dir)

    config = {
        "model": "MobileNetV2",
        "model_type": "11-class tomato verifier",
        "input_shape": [224, 224, 3],
        "input_range": [0, 1],
        "backbone_pretrained": "ImageNet",
        "backbone_trainable": False,
        "batch_size": 16,
        "epochs_max": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "optimizer": "Adam",
        "loss": "sparse_categorical_crossentropy",
        "class_names": class_names,
        "class_weights": class_weights,
        "hardware": "Windows CPU",
        "seed": SEED
    }

    with (run_dir / "training_config.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(config, f, indent=4)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(run_dir / "best_verifier.keras"),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            str(run_dir / "keras_training_log.csv")
        )
    ]

    print("\nStarting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    save_history(history, run_dir)

    # Save the restored best-validation-weight model.
    model.save(run_dir / "final_verifier.keras")

    best_epoch = (
        int(min(
            range(len(history.history["val_loss"])),
            key=lambda i: history.history["val_loss"][i]
        )) + 1
    )

    summary = {
        "best_epoch": best_epoch,
        "epochs_completed": len(history.history["loss"]),
        "best_validation_loss": float(min(history.history["val_loss"])),
        "best_validation_accuracy": float(
            max(history.history["val_accuracy"])
        ),
        "best_model": "best_verifier.keras",
        "test_evaluation": "Not performed in this training script"
    }

    with (run_dir / "training_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=4)

    print("\n" + "=" * 55)
    print("VERIFIER TRAINING COMPLETED")
    print("=" * 55)
    print(json.dumps(summary, indent=4))
    print("\nResults saved to:", run_dir)


if __name__ == "__main__":
    main()
