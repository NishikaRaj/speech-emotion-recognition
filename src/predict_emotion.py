
import sys
import os
import numpy as np
import pandas as pd
import librosa
import joblib

# Load the trained model
MODEL_PATH = "models/svm_emotion_model.joblib"

model = joblib.load(MODEL_PATH)


def extract_features(audio_path):
    # Load audio
    audio, sr = librosa.load(audio_path, sr=22050, mono=True)

    # Extract 40 MFCC coefficients
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40,
        n_fft=2048,
        hop_length=512
    )

    # Calculate mean and standard deviation
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # Combine into 80 features
    features = np.concatenate([mfcc_mean, mfcc_std])

    columns = (
        [f"mfcc_{i:02d}" for i in range(1, 41)] +
        [f"mfcc_{i:02d}" for i in range(41, 81)]
    )

    return pd.DataFrame([features], columns=columns)


def predict_emotion(audio_path):
    if not os.path.exists(audio_path):
        print("Audio file not found!")
        return

    features = extract_features(audio_path)
    prediction = model.predict(features)[0]

    print("\n------------------------------")
    print(" PREDICTED EMOTION:", prediction.upper())
    print("------------------------------")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict_emotion.py path/to/audio.wav")
    else:
        predict_emotion(sys.argv[1])
