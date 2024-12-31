import librosa
import librosa.display
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

def create_melspectrogram(audio_path, output_path, sr=22050, n_mels=128, n_fft=2048, 
                         hop_length=512, duration=30, segment_duration=4):
    """
    Convert audio file to mel spectrogram and save as numpy array
    Segments the spectrogram into 4-second chunks
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, duration=duration, sr=sr)
        
        # Compute mel spectrogram
        mel_spect = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
        )
        mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
        
        # Calculate number of frames per segment
        frames_per_segment = int((segment_duration * sr) / hop_length)
        
        # Segment the spectrogram
        num_segments = mel_spect_db.shape[1] // frames_per_segment
        segments = []
        
        for i in range(num_segments):
            start_frame = i * frames_per_segment
            end_frame = start_frame + frames_per_segment
            segment = mel_spect_db[:, start_frame:end_frame]
            segments.append(segment)
            
        # Stack segments into a single array (n_segments, n_mels, time_frames)
        segments = np.stack(segments, axis=0)
        
        # Save as numpy array instead of image
        np.save(output_path.replace('.png', '.npy'), segments)
        
        return True
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return False

def process_all_audio_files(input_dir, output_dir, sr=22050, n_mels=128, n_fft=2048,
                          hop_length=512, duration=30, segment_duration=4):
    """
    Process all audio files in the input directory
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    for genre_dir in tqdm(list(input_path.iterdir()), desc="Processing genres"):
        if genre_dir.is_dir():
            output_genre_path = output_path / genre_dir.name
            output_genre_path.mkdir(parents=True, exist_ok=True)
            
            audio_files = list(genre_dir.glob('*.wav'))
            
            for audio_file in tqdm(audio_files, desc=f"Processing {genre_dir.name}"):
                output_file_path = output_genre_path / f"{audio_file.stem}.npy"
                success = create_melspectrogram(
                    str(audio_file), 
                    str(output_file_path),
                    sr, n_mels, n_fft, hop_length, duration, segment_duration
                )
                if not success:
                    print(f"Failed to process: {audio_file.name}")

def main():
    parser = argparse.ArgumentParser(description="Convert audio files to grayscale spectrogram images.")
    parser.add_argument("input_dir", help="Directory containing audio files")
    parser.add_argument("output_dir", help="Directory to save spectrogram images")
    parser.add_argument("--sr", type=int, default=22050, help="Sample rate")
    parser.add_argument("--n_mels", type=int, default=128, help="Number of mel bands")
    parser.add_argument("--n_fft", type=int, default=2048, help="FFT window size")
    parser.add_argument("--hop_length", type=int, default=512, help="Number of samples between successive frames")
    parser.add_argument("--duration", type=int, default=30, help="Duration of audio to process (in seconds)")
    parser.add_argument("--segment_duration", type=int, default=4, help="Duration of each segment (in seconds)")

    args = parser.parse_args()

    process_all_audio_files(args.input_dir, args.output_dir, args.sr, args.n_mels, args.n_fft, args.hop_length, args.duration, args.segment_duration)

if __name__ == "__main__":
    main() 