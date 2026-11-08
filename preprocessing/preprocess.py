"""Image preprocessing constants and single-image preprocessing."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
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
DISEASE_CLASS_NAMES = [name for name in CLASS_NAMES if name != "Not_Tomato"]
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

def load_and_preprocess_image(img_path):
    """Return RGB float32 image, shape (224,224,3), normalized to [0,1]."""
    img_path = Path(img_path)
    with Image.open(img_path) as opened:
        oriented = ImageOps.exif_transpose(opened)
        rgb = oriented.convert("RGB")
        resized = rgb.resize(IMG_SIZE, Image.Resampling.LANCZOS)
        array = np.asarray(resized, dtype=np.float32).copy()
    array /= np.float32(255.0)
    if array.shape != (224, 224, 3) or not np.isfinite(array).all():
        raise ValueError(f"Invalid image after preprocessing: {img_path}")
    return np.clip(array, 0.0, 1.0).astype(np.float32, copy=False)
