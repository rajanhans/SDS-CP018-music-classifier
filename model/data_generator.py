import numpy as np
import tensorflow as tf

class TimeSegmentedSpectrogramGenerator(tf.keras.utils.Sequence):
    def __init__(self, paths, labels, batch_size=32, dim=(128, 128), 
                 n_channels=1, n_classes=10, shuffle=True, augment=False):
        super().__init__()
        self.paths = paths
        self.labels = labels
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment
        self.indexes = np.arange(len(self.paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        return int(np.floor(len(self.paths) / self.batch_size))
    
    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Generate data
        X, y = self.__data_generation(indexes)
        
        return X, y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, indexes):
        # Load data
        X = []
        y = []
        
        for idx in indexes:
            try:
                # Load saved numpy array (contains multiple time segments)
                segments = np.load(self.paths[idx])
                
                # Debug print
                #print(f"Original segments shape: {segments.shape}")
                
                # Resize segments to match expected dimensions
                resized_segments = []
                for segment in segments:
                    # Add channel dimension if needed for resize operation
                    if len(segment.shape) == 2:
                        segment = np.expand_dims(segment, axis=-1)
                    
                    # Debug print
                    #print(f"Segment shape before resize: {segment.shape}")
                    
                    # Resize each segment to match the expected dim
                    resized_segment = tf.image.resize(segment, self.dim)
                    resized_segments.append(resized_segment)
                
                segments = np.stack(resized_segments)
                
                # Debug print
                #print(f"Final segments shape: {segments.shape}")
                
                if self.augment:
                    segments = self._augment_segments(segments)
                
                X.append(segments)
                y.append(self.labels[self.paths[idx]])
                
            except Exception as e:
                print(f"Error processing file: {self.paths[idx]}")
                print(f"Error message: {str(e)}")
                raise
        
        # Stack all segments
        X = np.stack(X, axis=0)
        
        # Add channel dimension if needed
        if self.n_channels == 1:
            X = np.expand_dims(X, axis=-1)
        
        # Final debug print
        #print(f"Batch X shape: {X.shape}")
        
        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
    
    def _augment_segments(self, segments):
        # Add time-sequence aware augmentations here
        # For example, small frequency shifts that are consistent across segments
        if np.random.random() < 0.5:
            segments = segments + np.random.normal(0, 0.01, segments.shape)
        return segments 