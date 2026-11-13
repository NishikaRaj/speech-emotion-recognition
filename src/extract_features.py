
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "metadata.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"

# Feature extraction settings
SAMPLE_RATE = 22050
N_MFCC = 40


def extract_mfcc(audio_path):
    """Extract 40 MFCCs and summarize each using mean and std."""

    signal, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=2048,
        hop_length=512
    )

    # Summarize each coefficient across audio frames.
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # 40 means + 40 standard deviations = 80 features.
    return np.concatenate([mfcc_mean, mfcc_std])


def main():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata not found: {METADATA_PATH}\n"
            "Run python src/prepare_dataset.py first."
        )

    df = pd.read_csv(METADATA_PATH)
    records = []
    errors = []

    print(f"Total recordings to process: {len(df)}")
    print("Extracting MFCC features. Please wait...\n")

    for index, row in df.iterrows():
        try:
            features = extract_mfcc(row["file_path"])

            record = {
                "filename": row["filename"],
                "actor_id": int(row["actor_id"]),
                "emotion": row["emotion"],
            }

            for i, value in enumerate(features):
                record[f"mfcc_{i + 1:02d}"] = float(value)

            records.append(record)

        except Exception as exc:
            errors.append((row["filename"], str(exc)))
            print(f"Could not process {row['filename']}: {exc}")

        completed = index + 1
        if completed % 100 == 0 or completed == len(df):
            print(f"Processed {completed}/{len(df)} recordings")

    if not records:
        raise RuntimeError("No audio features were extracted.")

    features_df = pd.DataFrame(records)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(OUTPUT_PATH, index=False)

    print("\nFeature extraction complete!")
    print(f"Successful recordings: {len(records)}")
    print(f"Failed recordings: {len(errors)}")
    print(f"Feature columns: {features_df.shape[1]}")
    print(f"Saved to: {OUTPUT_PATH}")

    if errors:
        print("\nCheck the messages above for failed recordings.")


if __name__ == "__main__":
    main()
