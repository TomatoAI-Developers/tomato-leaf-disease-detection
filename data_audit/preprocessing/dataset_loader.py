

from pathlib import Path
import tensorflow as tf

from .preprocess import IMG_SIZE, BATCH_SIZE, CLASS_NAMES
from .augmentation import get_train_augmentation

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_image_paths(split_name):
    split_dir = DATASET_ROOT / split_name

    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing dataset split: {split_dir}")

    actual_classes = {
        p.name for p in split_dir.iterdir() if p.is_dir()
    }

    if actual_classes != set(CLASS_NAMES):
        raise ValueError(
            f"Class folder mismatch in {split_name}"
        )

    paths, labels = [], []

    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = split_dir / class_name

        files = sorted(
            p for p in class_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if not files:
            raise ValueError(f"No images found in {class_dir}")

        paths.extend(str(p) for p in files)
        labels.extend([label] * len(files))

    print(f"{split_name}: collected {len(paths)} images")
    return paths, labels


def decode_image(path, label):
    image = tf.io.read_file(path)

    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)

    # Keep pixel values in [0,255] until augmentation is complete.
    image = tf.clip_by_value(image, 0.0, 255.0)

    return image, tf.cast(label, tf.int32)


def normalize_batch(images, labels):
    images = tf.clip_by_value(images, 0.0, 255.0)
    images = images / 255.0
    return images, labels


def load_datasets():

    def load_split(split_name, training=False):
        paths, labels = collect_image_paths(split_name)

        ds = tf.data.Dataset.from_tensor_slices(
            (paths, labels)
        )

        if training:
            ds = ds.shuffle(
                buffer_size=len(paths),
                seed=42,
                reshuffle_each_iteration=True
            )

        ds = ds.map(
            decode_image,
            num_parallel_calls=1,
            deterministic=True
        )

        ds = ds.batch(BATCH_SIZE)

        if training:
            augmentation = get_train_augmentation()

            def augment_and_normalize(images, batch_labels):
                images = augmentation(images, training=True)
                return normalize_batch(images, batch_labels)

            ds = ds.map(
                augment_and_normalize,
                num_parallel_calls=1,
                deterministic=True
            )
        else:
            ds = ds.map(
                normalize_batch,
                num_parallel_calls=1,
                deterministic=True
            )

        return ds.prefetch(1)

    return (
        load_split("train", training=True),
        load_split("validation", training=False),
        load_split("test", training=False),
        CLASS_NAMES.copy()
    )

