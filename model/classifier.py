import tensorflow as tf
from tensorflow.keras import layers, Model
from .cnn import create_cnn_model
from .attention import MultiHeadSelfAttention

def create_music_genre_classifier(
    input_shape,
    num_classes,
    embed_dim,
    num_heads,
    num_transformer_blocks
):
    """
    Create a music genre classifier with CNN, Multi-head Attention, and Classification layers
    """
    
    # CNN Feature Extractor
    cnn_model = create_cnn_model(input_shape)
    
    # Input Layer
    inputs = layers.Input(shape=input_shape)
    
    # Apply CNN
    x = cnn_model(inputs)
    
    # Reshape for attention mechanism
    x = layers.Reshape((-1, x.shape[-1]))(x)
    
    # Position Embedding
    x = layers.Dense(embed_dim)(x)
    
    # Multi-head Attention Blocks
    for _ in range(num_transformer_blocks):
        x = MultiHeadSelfAttention(embed_dim, num_heads)(x)
    
    # Global Average Pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Classification Head
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model 