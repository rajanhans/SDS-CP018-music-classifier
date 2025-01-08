import streamlit as st
import librosa
import numpy as np
import pandas as pd
from joblib import load

# Load the pre-trained XGBoost model, scaler, and label encoder
xgb_model = load('genre_class_model_xgboost.joblib')
scaler = load('scaler.joblib')
label_encoder = load('label_encoder.joblib')

# Define the genre names corresponding to the encoded labels
genre_names = ['blues','classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']  

# Function to extract features from a wav file
def extract_features_from_wav(wav_file):
    # Load audio file using librosa
    y, sr = librosa.load(wav_file, sr=None)  # sr=None preserves the original sample rate
    
    # Extract various audio features
    
    # Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_bandwidth_mean = np.mean(spectral_bandwidth)
    
    # Chroma STFT (Short-Time Fourier Transform)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_stft_mean = np.mean(chroma_stft)
    chroma_stft_var = np.var(chroma_stft)
    
    # MFCCs (Mel-frequency cepstral coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=6)
    mfcc1_mean = np.mean(mfcc[0])
    mfcc1_var = np.var(mfcc[0])
    mfcc4_mean = np.mean(mfcc[3])
    mfcc5_var = np.var(mfcc[4])
    mfcc6_mean = np.mean(mfcc[5])
    
    # Roll-off (Spectral roll-off point)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    rolloff_mean = np.mean(rolloff)
    rolloff_var = np.var(rolloff)
    
    # Root Mean Square (RMS)
    rms = librosa.feature.rms(y=y)
    rms_var = np.var(rms)
    
    # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_var = np.var(spectral_centroid)
    
    # Tempo (beats per minute)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    # Harmony (harmonic-to-noise ratio)
    harmony = librosa.effects.harmonic(y)
    harmony_mean = np.mean(harmony)
    
    # **Perceptual Variance** (you may need to define this based on your specific requirement)
    perceptr_var = np.var(rms)  # Using RMS variance as a proxy for perceptual variance
    
    # Combine all extracted features
    features = [
        perceptr_var, spectral_bandwidth_mean, chroma_stft_mean, mfcc4_mean, chroma_stft_var, 
        mfcc1_var, rolloff_var, rms_var, rolloff_mean, mfcc1_mean, 
        spectral_centroid_var, mfcc5_var, mfcc6_mean, tempo, harmony_mean
    ]
    
    return features

# Streamlit App
def main():
    st.title("Music Genre Prediction from WAV File")
    
    st.write("""
    This Streamlit app allows you to upload a .wav file and predicts its genre based on the features extracted from the audio.
    """)
    
    # Upload WAV file
    uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])
    
    if uploaded_file is not None:
        # Extract features from the uploaded file
        st.write("Extracting features...")
        extracted_features = extract_features_from_wav(uploaded_file)
        
        # Convert extracted features into a DataFrame
        reduced_feature_columns = [
            "perceptr_var", "spectral_bandwidth_mean", "chroma_stft_mean", "mfcc4_mean", 
            "chroma_stft_var", "mfcc1_var", "rolloff_var", "rms_var", "rolloff_mean", 
            "mfcc1_mean", "spectral_centroid_var", "mfcc5_var", "mfcc6_mean", "tempo", 
            "harmony_mean"
        ]
        extracted_features_df = pd.DataFrame([extracted_features], columns=reduced_feature_columns)
        
        # Scale the extracted features using the saved scaler
        X_scaled = scaler.transform(extracted_features_df)
        
        # Predict the genre using the trained model
        predicted_label_encoded = xgb_model.predict(X_scaled)
        
        # Decode the predicted label
        predicted_label = label_encoder.inverse_transform(predicted_label_encoded)
        
        # Map the encoded label to the actual genre name
        predicted_genre_name = genre_names[predicted_label[0]]  # Map the predicted label to the actual genre name
        
        # Display the predicted genre
        st.write(f"The predicted genre of the song is: **{predicted_genre_name}**")

# Run the Streamlit app
if __name__ == "__main__":
    main()
