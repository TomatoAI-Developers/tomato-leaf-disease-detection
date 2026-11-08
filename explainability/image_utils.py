import base64
import io
from PIL import Image


def image_to_data_url(image, max_dimension=640, quality=82):
    """
    Convert a PIL image into a compressed JPEG Base64 data URL.
    Returns None if conversion fails.
    """
    try:
        image = image.convert("RGB")
        image.thumbnail(
            (max_dimension, max_dimension),
            Image.Resampling.LANCZOS
        )

        buffer = io.BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=quality,
            optimize=True
        )

        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return "data:image/jpeg;base64," + encoded

    except Exception as error:
        print("Image encoding skipped:", error)
        return None
