
from pathlib import Path
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "RAVDESS"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "metadata.csv"

# RAVDESS emotion codes
EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

def build_metadata():
    records = []

    for audio_path in sorted(DATASET_PATH.rglob("*.wav")):
        # RAVDESS filename format:
        # modality-vocal_channel-emotion-intensity-statement-repetition-actor.wav
        parts = audio_path.stem.split("-")

        if len(parts) != 7:
            print(f"Skipping unexpected filename: {audio_path.name}")
            continue

        emotion_code = parts[2]
        actor_id = parts[6]

        if emotion_code not in EMOTIONS:
            print(f"Skipping unknown emotion: {audio_path.name}")
            continue

        records.append({
            "file_path": str(audio_path.resolve()),
            "filename": audio_path.name,
            "actor_id": int(actor_id),
            "emotion_code": emotion_code,
            "emotion": EMOTIONS[emotion_code],
        })

    if not records:
        raise RuntimeError(
            f"No valid WAV files found under {DATASET_PATH}"
        )

    df = pd.DataFrame(records)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nTotal recordings: {len(df)}")
    print("\nRecordings per emotion:")
    print(df["emotion"].value_counts().sort_index())
    print(f"\nMetadata saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    build_metadata()
