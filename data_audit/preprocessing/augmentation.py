import tensorflow as tf

def get_train_augmentation():
    """
    Returns a sequential augmentation pipeline.
    Applied ONLY to training data.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),      
        tf.keras.layers.RandomZoom(0.1),          
        tf.keras.layers.RandomTranslation(0.1, 0.1), 
        tf.keras.layers.RandomBrightness(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])