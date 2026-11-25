
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "metadata.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"

SAMPLE_RATE = 22050
N_MFCC = 40


def extract_features(audio_path):
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

    rms = librosa.feature.rms(
        y=signal,
        frame_length=2048,
        hop_length=512
    )

    zcr = librosa.feature.zero_crossing_rate(
        y=signal,
        frame_length=2048,
        hop_length=512
    )

    f0, voiced_flag, voiced_prob = librosa.pyin(
        signal,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        frame_length=2048,
        hop_length=512
    )

    valid_pitch = f0[~np.isnan(f0)]

    if len(valid_pitch) > 0:
        pitch_features = [
            np.mean(valid_pitch),
            np.std(valid_pitch),
            np.min(valid_pitch),
            np.max(valid_pitch)
        ]
    else:
        pitch_features = [0, 0, 0, 0]

    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        [np.mean(rms), np.std(rms)],
        [np.mean(zcr), np.std(zcr)],
        pitch_features
    ])

    return features


def main():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata not found: {METADATA_PATH}")

    df = pd.read_csv(METADATA_PATH)
    records = []
    errors = []

    print(f"Total recordings: {len(df)}")
    print("Extracting MFCC, RMS, ZCR and pitch features...\n")

    for index, row in df.iterrows():
        try:
            features = extract_features(row["file_path"])

            record = {
                "filename": row["filename"],
                "actor_id": int(row["actor_id"]),
                "emotion": row["emotion"]
            }

            for i, value in enumerate(features[:80]):
                record[f"mfcc_{i + 1:02d}"] = float(value)

            record["rms_mean"] = float(features[80])
            record["rms_std"] = float(features[81])
            record["zcr_mean"] = float(features[82])
            record["zcr_std"] = float(features[83])

            record["pitch_mean"] = float(features[84])
            record["pitch_std"] = float(features[85])
            record["pitch_min"] = float(features[86])
            record["pitch_max"] = float(features[87])

            records.append(record)

        except Exception as exc:
            errors.append((row["filename"], str(exc)))
            print(f"Could not process {row['filename']}: {exc}")

        completed = index + 1
        if completed % 100 == 0 or completed == len(df):
            print(f"Processed {completed}/{len(df)}")

    if not records:
        raise RuntimeError("No features were extracted.")

    features_df = pd.DataFrame(records)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(OUTPUT_PATH, index=False)

    print("\nFeature extraction complete!")
    print("Successful recordings:", len(records))
    print("Failed recordings:", len(errors))
    print("Dataset shape:", features_df.shape)
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()

