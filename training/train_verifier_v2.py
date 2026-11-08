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

SEED = 42
HEAD_EPOCHS = 5
FINE_TUNE_EPOCHS = 10
FINE_TUNE_LAYERS = 30
HEAD_LR = 0.0001
FINE_TUNE_LR = 0.00001

tf.keras.utils.set_random_seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except RuntimeError:
    pass


def create_run_directory():
    parent = ROOT / "artifacts" / "phase3" / "verifier_v2"
    parent.mkdir(parents=True, exist_ok=True)

    while True:
        run_dir = parent / (
            "run_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            continue


def load_class_weights():
    if not WEIGHTS_FILE.is_file():
        raise FileNotFoundError(WEIGHTS_FILE)

    with WEIGHTS_FILE.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    weights = {int(k): float(v) for k, v in raw.items()}

    if set(weights) != set(range(len(CLASS_NAMES))):
        raise ValueError("Class weights do not match class mapping.")

    return weights


def build_model():
    inputs = tf.keras.Input(shape=(224, 224, 3), name="image")

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

    model = tf.keras.Model(inputs, outputs, name="TomatoAI_Verifier_V2")

    return model, backbone


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(
                name="accuracy"
            )
        ],
        weighted_metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(
                name="weighted_accuracy"
            )
        ]
    )


def make_callbacks(checkpoint_path, csv_path, patience):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
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
        tf.keras.callbacks.CSVLogger(str(csv_path))
    ]


def save_history(history, run_dir):
    with (run_dir / "training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "epoch"] + list(history[0].keys()))

        for stage, stage_history in enumerate(history, start=1):
            count = len(stage_history["loss"])

            for i in range(count):
                writer.writerow(
                    [stage, i + 1] +
                    [stage_history[key][i] for key in stage_history]
                )

    plt.figure(figsize=(9, 5))

    for stage, stage_history in enumerate(history, start=1):
        plt.plot(
            stage_history["val_accuracy"],
            label=f"Stage {stage} validation"
        )

    plt.xlabel("Epoch within stage")
    plt.ylabel("Validation accuracy")
    plt.title("TomatoAI Verifier V2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(run_dir / "validation_accuracy.png", dpi=150)
    plt.close()


def main():
    print("=" * 60)
    print("TomatoAI - Verifier V2: Two-Stage Training")
    print("=" * 60)

    if len(CLASS_NAMES) != 11:
        raise ValueError("Expected 11 verifier classes.")

    class_weights = load_class_weights()

    print("\nLoading datasets...")
    train_ds, val_ds, _, class_names = load_datasets()

    if class_names != CLASS_NAMES:
        raise ValueError("Class mapping mismatch.")

    print("\nTraining batches:", len(train_ds))
    print("Validation batches:", len(val_ds))

    print("\nBuilding MobileNetV2...")
    model, backbone = build_model()

    run_dir = create_run_directory()
    print("\nNew experiment directory:", run_dir)

    config = {
        "model": "MobileNetV2",
        "experiment": "Verifier V2",
        "class_names": CLASS_NAMES,
        "input_shape": [224, 224, 3],
        "input_range": [0, 1],
        "batch_size": 16,
        "augmentation": "mild",
        "stage_1_epochs": HEAD_EPOCHS,
        "stage_1_learning_rate": HEAD_LR,
        "stage_2_max_epochs": FINE_TUNE_EPOCHS,
        "stage_2_learning_rate": FINE_TUNE_LR,
        "fine_tune_layers": FINE_TUNE_LAYERS,
        "class_weights": class_weights,
        "test_evaluation": "Not performed during training",
        "seed": SEED
    }

    with (run_dir / "training_config.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(config, f, indent=4)

    histories = []

    # Stage 1: train classification head.
    print("\n" + "=" * 60)
    print("STAGE 1: TRAIN CLASSIFICATION HEAD")
    print("=" * 60)

    compile_model(model, HEAD_LR)

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=HEAD_EPOCHS,
        class_weight=class_weights,
        callbacks=make_callbacks(
            run_dir / "best_stage1.keras",
            run_dir / "stage1_log.csv",
            patience=3
        ),
        verbose=1
    )

    histories.append(history1.history)

    # Restore the best stage-1 checkpoint before fine-tuning.
    model = tf.keras.models.load_model(
        run_dir / "best_stage1.keras"
    )

    backbone = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "mobilenet" in layer.name.lower():
            backbone = layer
            break

    if backbone is None:
        raise RuntimeError("Could not locate MobileNetV2 backbone.")

    # Stage 2: unfreeze only upper backbone layers.
    print("\n" + "=" * 60)
    print("STAGE 2: FINE-TUNE UPPER MOBILENETV2 LAYERS")
    print("=" * 60)

    backbone.trainable = True

    for layer in backbone.layers:
        layer.trainable = False

    for layer in backbone.layers[-FINE_TUNE_LAYERS:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    trainable_count = sum(
        int(tf.keras.backend.count_params(w))
        for w in model.trainable_weights
    )

    print("Trainable parameters during fine-tuning:", trainable_count)

    compile_model(model, FINE_TUNE_LR)

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINE_TUNE_EPOCHS,
        class_weight=class_weights,
        callbacks=make_callbacks(
            run_dir / "best_stage2.keras",
            run_dir / "stage2_log.csv",
            patience=4
        ),
        verbose=1
    )

    histories.append(history2.history)

    # Select the better checkpoint using validation loss.
    stage1_best = tf.keras.models.load_model(
        run_dir / "best_stage1.keras"
    )
    stage2_best = tf.keras.models.load_model(
        run_dir / "best_stage2.keras"
    )

    stage1_loss = float(stage1_best.evaluate(val_ds, verbose=0)[0])
    stage2_loss = float(stage2_best.evaluate(val_ds, verbose=0)[0])

    if stage2_loss < stage1_loss:
        best_model = stage2_best
        selected_stage = 2
        selected_loss = stage2_loss
    else:
        best_model = stage1_best
        selected_stage = 1
        selected_loss = stage1_loss

    best_model.save(run_dir / "best_verifier_v2.keras")

    save_history(histories, run_dir)

    summary = {
        "selected_stage": selected_stage,
        "selected_validation_loss": selected_loss,
        "stage1_validation_loss": stage1_loss,
        "stage2_validation_loss": stage2_loss,
        "stage1_epochs_completed": len(history1.history["loss"]),
        "stage2_epochs_completed": len(history2.history["loss"]),
        "best_model": "best_verifier_v2.keras",
        "test_evaluation": "Not performed"
    }

    with (run_dir / "training_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=4)

    print("\n" + "=" * 60)
    print("VERIFIER V2 TRAINING COMPLETED")
    print("=" * 60)
    print(json.dumps(summary, indent=4))
    print("\nResults saved to:", run_dir)


if __name__ == "__main__":
    main()
