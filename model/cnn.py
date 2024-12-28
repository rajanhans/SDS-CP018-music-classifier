import tensorflow as tf
from tensorflow.keras import layers

def create_cnn_model(input_shape):
    """
    Create a CNN model for feature extraction from spectrograms.
    """
    cnn_model = tf.keras.Sequential([
        # First Conv Block
        layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),
        
        # Second Conv Block
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),
        
        # Third Conv Block
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),
        
        # Fourth Conv Block
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),
    ])
    
    return cnn_model 