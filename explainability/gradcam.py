import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps


def generate_gradcam(model, image_array, class_index, alpha=0.40):
    """
    Generate a Grad-CAM overlay using the existing trained MobileNetV2 model.

    Args:
        model: Loaded TomatoAI MobileNetV2 model.
        image_array: Preprocessed image, shape (1, 224, 224, 3),
                     with pixel values in the range 0 to 1.
        class_index: Class index whose prediction is being explained.
        alpha: Heatmap overlay strength.

    Returns:
        PIL RGB image containing the Grad-CAM overlay,
        or None if a valid heatmap cannot be generated.
    """

    try:
        image_array = np.asarray(image_array, dtype=np.float32)

        if image_array.shape != (1, 224, 224, 3):
            return None

        if not np.all(np.isfinite(image_array)):
            return None

        if not 0 <= int(class_index) < 11:
            return None

        # Reuse the original trained layers.
        backbone = model.get_layer("mobilenetv2_1.00_224")
        last_conv = backbone.get_layer("out_relu")

        normalization = model.get_layer("mobilenet_normalization")
        gap = model.get_layer("global_average_pooling2d")
        dropout = model.get_layer("dropout")
        classifier = model.get_layer("verifier_predictions")

        # Extract the final convolutional feature maps.
        feature_model = tf.keras.Model(
            inputs=backbone.input,
            outputs=last_conv.output
        )

        input_tensor = tf.convert_to_tensor(image_array)

        with tf.GradientTape() as tape:
            normalized = normalization(input_tensor, training=False)

            features = feature_model(
                normalized,
                training=False
            )

            pooled = gap(features)
            dropped = dropout(pooled, training=False)
            predictions = classifier(dropped)

            class_score = predictions[0, int(class_index)]

        gradients = tape.gradient(class_score, features)

        if gradients is None:
            return None

        if not bool(
            tf.reduce_all(tf.math.is_finite(gradients)).numpy()
        ):
            return None

        # Calculate Grad-CAM channel weights.
        channel_weights = tf.reduce_mean(
            gradients,
            axis=(1, 2)
        )

        # Combine feature maps using the gradient weights.
        cam = tf.reduce_sum(
            features[0] * channel_weights[0],
            axis=-1
        )

        # Keep positive contributions only.
        cam = tf.nn.relu(cam)

        maximum = tf.reduce_max(cam)

        if not np.isfinite(float(maximum.numpy())):
            return None

        if float(maximum.numpy()) <= 1e-10:
            return None

        cam = cam / maximum
        heatmap = cam.numpy()

        # Resize the 7x7 heatmap to the model input dimensions.
        heatmap_image = Image.fromarray(
            np.uint8(np.clip(heatmap, 0, 1) * 255)
        )

        heatmap_image = heatmap_image.resize(
            (224, 224),
            Image.Resampling.BILINEAR
        )

        # Apply a blue-to-red colour scale.
        colored_heatmap = ImageOps.colorize(
            heatmap_image,
            black="#000080",
            white="#FF0000"
        ).convert("RGB")

        # Prepare the original preprocessed image.
        original = Image.fromarray(
            np.uint8(
                np.clip(image_array[0], 0, 1) * 255
            )
        ).convert("RGB")

        # Blend the heatmap with the original image.
        overlay = Image.blend(
            original,
            colored_heatmap,
            alpha=float(np.clip(alpha, 0, 1))
        )

        return overlay

    except Exception as error:
        print("Grad-CAM generation skipped:", error)
        return None
