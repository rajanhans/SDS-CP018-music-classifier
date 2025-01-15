import streamlit as st
import numpy as np
import librosa
import joblib
from keras.models import load_model
import tempfile

# -------------------------------------------------------------------------
# 1) Feature Extraction (same 57 features you used in training)
#    This function should match exactly how you extracted features
#    in your XGBoost/CNN training scripts.
# -------------------------------------------------------------------------
def extract_all_features_from_wav(wav_file):
    """
    Extract 57 features based on your training code:
    (1) Chroma STFT mean,var
    (2) RMS mean,var
    (3) Spectral Centroid mean,var
    (4) Spectral Bandwidth mean,var
    (5) Spectral Rolloff mean,var
    (6) Zero Crossing Rate mean,var
    (7) Harmony mean,var
    (8) Perceptual mean,var
    (9) Tempo
    (10) 20 MFCC means + 20 MFCC vars
    """
    try:
        # Load audio (force mono)
        y, sr = librosa.load(wav_file, sr=None, mono=True)

        # 1. Spectral Bandwidth
        sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        sb_mean, sb_var = np.mean(sb), np.var(sb)

        # 2. Chroma STFT
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean, chroma_var = np.mean(chroma), np.var(chroma)

        # 3. RMS
        rms = librosa.feature.rms(y=y)
        rms_mean, rms_var = np.mean(rms), np.var(rms)

        # 4. Spectral Centroid
        sc = librosa.feature.spectral_centroid(y=y, sr=sr)
        sc_mean, sc_var = np.mean(sc), np.var(sc)

        # 5. Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
        rolloff_mean, rolloff_var = np.mean(rolloff), np.var(rolloff)

        # 6. Zero-Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y=y)
        zcr_mean, zcr_var = np.mean(zcr), np.var(zcr)

        # 7. Harmony
        harmony = librosa.effects.harmonic(y)
        harmony_mean, harmony_var = np.mean(harmony), np.var(harmony)

        # 8. Perceptual (using RMS)
        perceptr_mean, perceptr_var = rms_mean, rms_var

        # 9. Tempo (convert from shape (1,) to scalar float)
        tempo_array, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo_array)

        # 10. MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_means = [np.mean(mfcc[i]) for i in range(20)]
        mfcc_vars = [np.var(mfcc[i]) for i in range(20)]

        features = [
            chroma_mean, chroma_var,
            rms_mean, rms_var,
            sc_mean, sc_var,
            sb_mean, sb_var,
            rolloff_mean, rolloff_var,
            zcr_mean, zcr_var,
            harmony_mean, harmony_var,
            perceptr_mean, perceptr_var,
            tempo
        ] + mfcc_means + mfcc_vars

        return np.array(features)

    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

# -------------------------------------------------------------------------
# 2) Streamlit App
# -------------------------------------------------------------------------
def main():
    st.title("Music Genre Classification")

    # File uploader for .wav
    uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

    # Radio button: user chooses XGBoost or CNN
    model_choice = st.radio(
        "Select your model:",
        ("XGBoost", "CNN")
    )

    # Predict button
    if st.button("Predict Genre"):
        if uploaded_file is None:
            st.warning("Please upload a WAV file.")
            return

        # Write the uploaded WAV to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_filename = tmp.name

        # Extract 57 features
        features = extract_all_features_from_wav(tmp_filename)
        if features is None:
            return  # We already showed an error if extraction failed

        # Reshape to (1, 57)
        features_reshaped = features.reshape(1, -1)

        # -------------------------------------
        # If user chooses XGBoost
        # -------------------------------------
        if model_choice == "XGBoost":
            try:
                # Load XGBoost model, scaler, label encoder
                xgb_model = joblib.load("notebooks/rajan-hans/genre_model_xgboost_full.joblib")
                xgb_scaler = joblib.load("notebooks/rajan-hans/scaler.joblib")
                xgb_label_encoder = joblib.load("notebooks/rajan-hans/label_encoder.joblib")
            except Exception as e:
                st.error(f"Error loading XGBoost model/scaler/encoder: {e}")
                return

            # Scale
            try:
                features_scaled = xgb_scaler.transform(features_reshaped)
            except Exception as e:
                st.error(f"Error scaling features for XGBoost: {e}")
                return

            # Predict
            try:
                prediction = xgb_model.predict(features_scaled)
                predicted_genre = xgb_label_encoder.inverse_transform(prediction)[0]
                st.success(f"Predicted Genre (XGBoost): {predicted_genre}")
            except Exception as e:
                st.error(f"Error predicting with XGBoost: {e}")

        # -------------------------------------
        # If user chooses CNN
        # -------------------------------------
        else:  # CNN
            try:
                # Load CNN model (Keras .h5), joblib scalers, joblib label encoder
                cnn_model = load_model("notebooks/rajan-hans/music_genre_cnn_model.h5")
                cnn_scaler = joblib.load("notebooks/rajan-hans/scaler_cnn.joblib")
                cnn_label_encoder = joblib.load("notebooks/rajan-hans/label_encoder_cnn.joblib")
            except Exception as e:
                st.error(f"Error loading CNN model/scaler/encoder: {e}")
                return

            # Scale
            try:
                features_scaled = cnn_scaler.transform(features_reshaped)
            except Exception as e:
                st.error(f"Error scaling features for CNN: {e}")
                return

            # Reshape for Conv1D: (1, 57, 1)
            try:
                features_scaled_cnn = features_scaled.reshape((1, features_scaled.shape[1], 1))
            except Exception as e:
                st.error(f"Error reshaping for CNN input: {e}")
                return

            # Predict
            try:
                y_pred = cnn_model.predict(features_scaled_cnn)
                predicted_class_idx = np.argmax(y_pred, axis=1)
                predicted_genre = cnn_label_encoder.inverse_transform(predicted_class_idx)[0]
                st.success(f"Predicted Genre (CNN): {predicted_genre}")
            except Exception as e:
                st.error(f"Error predicting with CNN: {e}")

if __name__ == "__main__":
    main()
