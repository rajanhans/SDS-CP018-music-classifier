import tensorflow as tf
import numpy as np

class SpectrogramSequenceGenerator(tf.keras.utils.Sequence):
    """
    Data generator for loading and preprocessing spectrogram segments
    """
    def __init__(self, spectrogram_paths, labels, batch_size=32, 
                 dim=(128, 128), n_channels=1, n_classes=10, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.spectrogram_paths = spectrogram_paths
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
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

    def __data_generation(self, batch_paths):
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        for i, path in enumerate(batch_paths):
            img = tf.keras.preprocessing.image.load_img(
                path, 
                color_mode='grayscale',
                target_size=self.dim
            )
            X[i,] = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            y[i] = self.labels[path]

        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes) 