
import os
import glob
import numpy as np
import librosa

# ============================================================
# 1. SETTINGS
# ============================================================

DATA_PATH = "data/raw/RAVDESS"
OUTPUT_PATH = "data/processed/mel_spectrograms.npz"

SAMPLE_RATE = 22050
DURATION = 4
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


# ============================================================
# 2. EXTRACT MEL SPECTROGRAM
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
# 3. PREPARE DATASET
# ============================================================

def main():

    # IMPORTANT: Always sort files to preserve identical order
    # in prepare_spectrograms.py and train_cnn.py.

    audio_files = sorted(
        glob.glob(
            os.path.join(DATA_PATH, "**", "*.wav"),
            recursive=True
        )
    )

    print("Total audio files found:", len(audio_files))

    if len(audio_files) == 0:
        raise FileNotFoundError(
            f"No WAV files found inside {DATA_PATH}"
        )

    X = []
    y = []
    filenames = []

    failed = 0

    for i, audio_path in enumerate(audio_files):

        filename = os.path.basename(audio_path)

        parts = filename.split("-")

        if len(parts) != 7:
            print("Skipping invalid filename:", filename)
            continue

        emotion_code = parts[2]
        emotion = EMOTION_MAP.get(emotion_code)

        if emotion is None:
            continue

        try:

            mel = extract_mel_spectrogram(audio_path)

            X.append(mel)
            y.append(emotion)
            filenames.append(filename)

        except Exception as e:

            failed += 1
            print("Failed:", filename, e)

        if (i + 1) % 100 == 0:
            print(
                f"Processed {i + 1}/{len(audio_files)}"
            )

    if len(X) == 0:
        raise ValueError("No spectrograms were generated.")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    filenames = np.asarray(filenames)

    # Add CNN channel dimension
    X = X[..., np.newaxis]

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    np.savez_compressed(
        OUTPUT_PATH,
        X=X,
        y=y,
        filenames=filenames
    )

    print("\n===================================")
    print("SPECTROGRAM PREPARATION COMPLETED")
    print("===================================")

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Total processed:", len(X))
    print("Failed files:", failed)
    print("Emotion distribution:")

    for emotion in sorted(EMOTION_MAP.values()):
        print(
            emotion + ":",
            np.sum(y == emotion)
        )

    print("\nSaved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
