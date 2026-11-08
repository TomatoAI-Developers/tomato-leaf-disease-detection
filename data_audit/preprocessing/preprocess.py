import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# --- Global Constants ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}

# Deterministic alphabetical ordering of the 11 original folders
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
    "Tomato_Yellow_Leaf_CurlVirus"
]

# 10-class mapping for disease classifier (excludes Not_Tomato, indexed 0-9)
DISEASE_CLASS_NAMES = [cls for cls in CLASS_NAMES if cls != "Not_Tomato"]

def load_and_preprocess_image(img_path):
    """
    Loads a single image, corrects EXIF orientation, converts to RGB,
    resizes to IMG_SIZE, and normalizes to [0, 1].
    Preserved for compatibility with other project modules.
    """
    # Context manager ensures the file handle is closed reliably
    with Image.open(img_path) as img:
        # Robust EXIF Orientation correction using PIL ImageOps
        img = ImageOps.exif_transpose(img)
            
        # Convert to RGB (handles grayscale and alpha channels)
        img = img.convert("RGB")
        
        # Resize
        img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0
        
    return img_array