import tensorflow as tf
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from model.classifier import create_music_genre_classifier
from model.data_generator import SpectrogramSequenceGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def prepare_data(spectrograms_dir):
    """
    Prepare data paths and labels from the spectrograms directory
    """
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
              'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    paths = []
    labels = {}
    
    for idx, genre in enumerate(genres):
        genre_path = Path(spectrograms_dir) / genre
        for spec_path in genre_path.glob('*.png'):
            paths.append(str(spec_path))
            labels[str(spec_path)] = idx
    
    return paths, labels, genres

def plot_training_history(history):
    """
    Plot training history
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def main():
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Parameters
    SPECTROGRAMS_DIR = '/Users/julienh/Desktop/SDS/SDS-CP018-music-classifier/Data/grayscale-Spectrograms'  # Update this path
    BATCH_SIZE = 32
    EPOCHS = 50
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    CHANNELS = 1
    
    # Prepare data
    print("Preparing data...")
    spectrogram_paths, labels, genres = prepare_data(SPECTROGRAMS_DIR)
    num_classes = len(genres)
    
    # Split data into train, validation, and test sets
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        spectrogram_paths,
        list(labels.values()),  # Pass the actual list of labels
        test_size=0.2,
        random_state=42,
        stratify=list(labels.values())  # Stratify by genre labels
    )
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths,
        train_labels,  # Use the corresponding labels for the second split
        test_size=0.2,
        random_state=42,
        stratify=train_labels  # Stratify again for validation set
    )
    
    print(f"Number of training samples: {len(train_paths)}")
    print(f"Number of validation samples: {len(val_paths)}")
    print(f"Number of test samples: {len(test_paths)}")
    
    # Create data generators
    train_generator = SpectrogramSequenceGenerator(
        train_paths,
        labels,  # Pass the labels dictionary here
        batch_size=BATCH_SIZE,
        dim=(IMG_HEIGHT, IMG_WIDTH),
        n_channels=CHANNELS,
        n_classes=num_classes,
        augment=True
    )

    val_generator = SpectrogramSequenceGenerator(
        val_paths,
        labels,  # Pass the labels dictionary here
        batch_size=BATCH_SIZE,
        dim=(IMG_HEIGHT, IMG_WIDTH),
        n_channels=CHANNELS,
        n_classes=num_classes
    )

    test_generator = SpectrogramSequenceGenerator(
        test_paths,
        labels,  # Pass the labels dictionary here
        batch_size=BATCH_SIZE,
        dim=(IMG_HEIGHT, IMG_WIDTH),
        n_channels=CHANNELS,
        n_classes=num_classes,
        shuffle=False
    )
    
    # Create and compile model
    print("Creating model...")
    model = create_music_genre_classifier(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS),
        num_classes=num_classes,
        embed_dim=256,
        num_heads=8,
        num_transformer_blocks=2
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=0.0001
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]
    
    # Train model
    print("Training model...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on test set
    print("\nEvaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Generate predictions for confusion matrix
    y_pred = []
    y_true = []
    
    for i in range(len(test_generator)):
        x, y = test_generator[i]
        pred = model.predict(x)
        y_pred.extend(np.argmax(pred, axis=1))
        y_true.extend(np.argmax(y, axis=1))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, genres)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=genres))
    
    # Save model
    model.save('final_model.keras')
    print("\nModel saved as 'final_model.keras'")

if __name__ == "__main__":
    main() 