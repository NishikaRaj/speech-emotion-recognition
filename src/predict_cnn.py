
import os
import numpy as np
import librosa
import tensorflow as tf

# ============================================================
# 1. SETTINGS
# ============================================================

MODEL_PATH = "models/cnn_emotion_model.keras"
LABEL_PATH = "models/cnn_labels.npy"

# This sample belongs to Actor_12
AUDIO_PATH = ( "data/raw/RAVDESS/Audio_Speech_Actors_01-24/Actor_16/03-01-05-01-02-01-16.wav"
)

SAMPLE_RATE = 22050
DURATION = 4
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512


# ============================================================
# 2. LOAD MODEL AND LABELS
# ============================================================

print("Loading CNN model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"CNN model not found: {MODEL_PATH}. "
        "Train the CNN first."
    )

if not os.path.exists(LABEL_PATH):
    raise FileNotFoundError(
        f"Emotion labels not found: {LABEL_PATH}. "
        "Train the CNN first."
    )

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

labels = np.load(LABEL_PATH, allow_pickle=True)

print("CNN model loaded successfully!")
print("Emotion labels:", labels)


# ============================================================
# 3. EXTRACT MEL SPECTROGRAM
# ============================================================

def extract_mel_spectrogram(audio_path):

    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    target_length = SAMPLE_RATE * DURATION

    if len(audio) < target_length:
        audio = np.pad(
            audio,
            (0, target_length - len(audio))
        )
    else:
        audio = audio[:target_length]

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max,
        top_db=80
    )

    mel_normalized = (mel_db + 80.0) / 80.0

    return mel_normalized.astype(np.float32)


# ============================================================
# 4. PREDICT EMOTION
# ============================================================

def predict_emotion(audio_path):

    if not os.path.exists(audio_path):
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    print("\nProcessing audio...")

    mel = extract_mel_spectrogram(audio_path)

    # Add batch and channel dimensions.
    # The model performs normalization internally.
    X = mel[np.newaxis, ..., np.newaxis]

    print("Input shape:", X.shape)

    expected_shape = tuple(model.input_shape[1:])

    if tuple(X.shape[1:]) != expected_shape:
        raise ValueError(
            f"Input shape mismatch! "
            f"Expected {expected_shape}, "
            f"got {X.shape[1:]}"
        )

    print("\nPredicting emotion...")

    predictions = model.predict(X, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))
    predicted_emotion = labels[predicted_index]

    score = predictions[predicted_index] * 100

    print("\n========== CNN RESULT ==========")
    print("Predicted Emotion:", predicted_emotion)
    print(f"Model score: {score:.2f}%")

    print("\nAll Emotion Scores:")

    for label, probability in zip(labels, predictions):
        print(f"{label}: {probability * 100:.2f}%")

    print("================================")


# ============================================================
# 5. RUN
# ============================================================

if __name__ == "__main__":
    predict_emotion(AUDIO_PATH)
