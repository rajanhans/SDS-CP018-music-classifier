import tensorflow as tf
import numpy as np

class SpectrogramSequenceGenerator(tf.keras.utils.Sequence):
    """
    Data generator for loading and preprocessing spectrogram segments
    """
    def __init__(self, spectrogram_paths, labels, batch_size=32, 
                 dim=(128, 128), n_channels=1, n_classes=10, shuffle=True, augment=False, **kwargs):
        """
        Initialize the data generator
        """
        super().__init__(**kwargs)  # Call the base class constructor
        self.spectrogram_paths = spectrogram_paths
        self.labels = labels  # This should now be a dictionary
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.spectrogram_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_paths = [self.spectrogram_paths[k] for k in indexes]
        X, y = self.__data_generation(batch_paths)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.spectrogram_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """
        Generates data containing batch_size samples
        """
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Load spectrogram
            img = self.load_spectrogram(ID)

            # Data augmentation
            if self.augment:
                img = self.data_augmentation(img)

            # Store sample
            X[i,] = img

            # Store class using the dictionary lookup
            y[i] = self.labels[ID]

        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)

    def data_augmentation(self, img):
        """
        Apply data augmentation to the image
        """
        # Example augmentations (you can add more)
        if np.random.rand() > 0.5:
            img = img[::-1, :, :]  # Flip horizontally
        if np.random.rand() > 0.5:
            img = img[:, ::-1, :]  # Flip vertically
        if np.random.rand() > 0.5:
            img = np.rot90(img)  # Rotate 90 degrees
        return img

    def load_spectrogram(self, spectrogram_path):
        """
        Load and preprocess the spectrogram from the given path
        """
        try:
            img = tf.io.read_file(spectrogram_path)
            img = tf.image.decode_png(img, channels=self.n_channels)
            img = tf.image.resize(img, self.dim)
            img = tf.cast(img, tf.float32) / 255.0  # Normalize to [0, 1]
        except Exception as e:
            print(f"Error loading spectrogram: {spectrogram_path} - {e}")
            # Return a placeholder or handle the error as appropriate for your application
            img = np.zeros((*self.dim, self.n_channels), dtype=np.float32)
        return img 