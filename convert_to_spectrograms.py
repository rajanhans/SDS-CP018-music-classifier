import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

# Default parameters for mel spectrogram
SR = 22050  # Sample rate
N_MELS = 128  # Number of mel bands
N_FFT = 2048  # FFT window size
HOP_LENGTH = 512  # Number of samples between successive frames
DURATION = 30  # Duration of audio to process (in seconds)

def create_melspectrogram(audio_path, output_path, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, duration=DURATION):
    """
    Convert audio file to mel spectrogram and save as grayscale image
    """
    try:
        y, sr = librosa.load(audio_path, duration=duration, sr=sr)

        # Mel spectrogram
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
        mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)

        # Create figure with clean background
        plt.figure(figsize=(10, 4))
        plt.axis('off')  # Turn off axis
        plt.margins(0,0)  # Remove margins
        
        # Display spectrogram without axes or grid
        librosa.display.specshow(
            mel_spect_db, 
            sr=sr, 
            hop_length=hop_length,
            cmap='gray',
            x_axis=None,
            y_axis=None
        )
        
        # Save with tight layout and no extra space
        plt.savefig(
            output_path,
            bbox_inches='tight',
            pad_inches=0,
            dpi=100,
            format='png',
            facecolor='black',
            edgecolor='none'
        )
        plt.close()

        return True
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return False

def process_all_audio_files(input_dir, output_dir, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, duration=DURATION):
    """
    Process all audio files in the input directory, create mel spectrograms, and save them as grayscale images
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    for genre_dir in tqdm(list(input_path.iterdir()), desc="Processing genres", unit="genre"):
        if genre_dir.is_dir():
            output_genre_path = output_path / genre_dir.name
            output_genre_path.mkdir(parents=True, exist_ok=True)

            audio_files = list(genre_dir.glob('*.wav'))  # Adjust the glob pattern for other audio formats

            for audio_file in tqdm(audio_files, desc=f"Processing {genre_dir.name} audio files", unit="file"):
                output_file_path = output_genre_path / f"{audio_file.stem}.png"
                success = create_melspectrogram(str(audio_file), str(output_file_path), sr, n_mels, n_fft, hop_length, duration)
                if not success:
                    print(f"Failed to process: {audio_file.name}")

def main():
    parser = argparse.ArgumentParser(description="Convert audio files to grayscale spectrogram images.")
    parser.add_argument("input_dir", help="Directory containing audio files")
    parser.add_argument("output_dir", help="Directory to save spectrogram images")
    parser.add_argument("--sr", type=int, default=SR, help="Sample rate")
    parser.add_argument("--n_mels", type=int, default=N_MELS, help="Number of mel bands")
    parser.add_argument("--n_fft", type=int, default=N_FFT, help="FFT window size")
    parser.add_argument("--hop_length", type=int, default=HOP_LENGTH, help="Number of samples between successive frames")
    parser.add_argument("--duration", type=int, default=DURATION, help="Duration of audio to process (in seconds)")

    args = parser.parse_args()

    process_all_audio_files(args.input_dir, args.output_dir, args.sr, args.n_mels, args.n_fft, args.hop_length, args.duration)

if __name__ == "__main__":
    main() 