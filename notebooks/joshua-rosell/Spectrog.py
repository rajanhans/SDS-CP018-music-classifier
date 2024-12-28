import os
import librosa
import numpy as np
import soundfile as sf
import audioread
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

# Path to the data folder
data_path = 'Data'
genres = os.listdir(data_path)

# Create directories for spectrograms
os.makedirs('Spectrograms/train', exist_ok=True)
os.makedirs('Spectrograms/test', exist_ok=True)

# Function to create spectrogram
def create_spectrogram(file_path, output_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_DB = librosa.power_to_db(S, ref=np.max)
        
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-frequency spectrogram')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    except sf.LibsndfileError as e:
        print(f"Error opening {file_path}: {e}")
    except audioread.NoBackendError:
        print(f"No backend found for {file_path}")

# Process each genre
for genre in genres:
    genre_path = os.path.join(data_path, genre)
    files = os.listdir(genre_path)
    
    # Split files into training and test sets
    train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
    
    # Create spectrograms for training set
    for file in train_files:
        file_path = os.path.join(genre_path, file)
        output_path = os.path.join('Spectrograms/train', f'{genre}_{file}.png')
        create_spectrogram(file_path, output_path)
    
    # Create spectrograms for test set
    for file in test_files:
        file_path = os.path.join(genre_path, file)
        output_path = os.path.join('Spectrograms/test', f'{genre}_{file}.png')
        create_spectrogram(file_path, output_path)

print("Spectrograms created and split into training and test sets.")

