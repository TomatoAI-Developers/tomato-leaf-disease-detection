"""Mild training-only augmentation for TomatoAI."""
import tensorflow as tf


def get_train_augmentation():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),

        # Rotation range: approximately +/- 7.2 degrees.
        tf.keras.layers.RandomRotation(
            0.02,
            fill_mode="reflect"
        ),

        tf.keras.layers.RandomZoom(
            0.05,
            fill_mode="reflect"
        ),

        tf.keras.layers.RandomTranslation(
            0.03,
            0.03,
            fill_mode="reflect"
        ),

        tf.keras.layers.RandomBrightness(
            0.05,
            value_range=(0.0, 255.0)
        ),

        tf.keras.layers.RandomContrast(0.05),

    ], name="mild_train_augmentation")
