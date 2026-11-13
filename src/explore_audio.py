
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "metadata.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "audio_exploration.png"


def explore_audio():
    # Load our dataset metadata
    df = pd.read_csv(METADATA_PATH)

    # Select one recording to study
    sample = df.iloc[0]
    audio_path = sample["file_path"]
    emotion = sample["emotion"]

    # Load audio at its original sample rate
    signal, sr = librosa.load(audio_path, sr=None)

    duration = len(signal) / sr

    print("Audio file:", sample["filename"])
    print("Emotion label:", emotion)
    print("Sample rate:", sr, "Hz")
    print("Duration:", round(duration, 2), "seconds")
    print("Audio samples:", len(signal))

    # Generate a Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_mels=128,
        n_fft=2048,
        hop_length=512
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Plot waveform and Mel spectrogram
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    librosa.display.waveshow(
        signal,
        sr=sr,
        ax=axes[0]
    )
    axes[0].set_title(f"Speech Waveform — {emotion}")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Amplitude")

    image = librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=512,
        x_axis="time",
        y_axis="mel",
        ax=axes[1]
    )
    axes[1].set_title("Mel Spectrogram")
    axes[1].set_xlabel("Time (seconds)")
    fig.colorbar(image, ax=axes[1], format="%+2.0f dB")

    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150)
    print("Plot saved to:", OUTPUT_PATH)

    plt.show()


if __name__ == "__main__":
    explore_audio()
