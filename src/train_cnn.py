
# ============================================================
# SPEECH EMOTION RECOGNITION - CNN MODEL TRAINING
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# 1. SETTINGS
# ============================================================

SPECTROGRAM_PATH = "data/processed/mel_spectrograms.npz"

MODEL_PATH = "models/cnn_emotion_model.keras"
LABELS_PATH = "models/cnn_labels.npy"

CONFUSION_MATRIX_PATH = "outputs/cnn_confusion_matrix.png"
HISTORY_PATH = "outputs/cnn_training_history.png"

RANDOM_STATE = 42
EPOCHS = 30
BATCH_SIZE = 32

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ============================================================
# 2. LOAD MEL SPECTROGRAM DATA
# ============================================================

print("\nLoading dataset...")

if not os.path.exists(SPECTROGRAM_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {SPECTROGRAM_PATH}. "
        "Run prepare_spectrograms.py first."
    )

data = np.load(SPECTROGRAM_PATH, allow_pickle=True)

X = data["X"].astype(np.float32)
y = np.asarray(data["y"]).astype(str)
filenames = np.asarray(data["filenames"]).astype(str)

print("Dataset loaded successfully!")
print("X shape:", X.shape)
print("y shape:", y.shape)

print("Minimum:", np.min(X))
print("Maximum:", np.max(X))
print("Mean:", np.mean(X))
print("Standard deviation:", np.std(X))


# ============================================================
# 3. VALIDATE DATASET
# ============================================================

if X.ndim == 3:
    X = X[..., np.newaxis]

if X.ndim != 4:
    raise ValueError(
        f"Expected 4D input, received {X.shape}"
    )

if len(X) != len(y) or len(X) != len(filenames):
    raise ValueError(
        f"Dataset mismatch: X={len(X)}, "
        f"y={len(y)}, filenames={len(filenames)}"
    )

if not np.isfinite(X).all():
    raise ValueError("Spectrogram data contains NaN or infinite values!")

print("\nDataset validation successful!")


# ============================================================
# 4. EXTRACT ACTOR IDS
# ============================================================

print("\nExtracting actor IDs...")

actor_ids = np.array([
    filename.split("-")[-1].replace(".wav", "")
    for filename in filenames
])

print("Total audio files:", len(filenames))
print("Unique actors:", len(np.unique(actor_ids)))


# ============================================================
# 5. ENCODE EMOTION LABELS
# ============================================================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

num_classes = len(encoder.classes_)

print("\nEmotion classes:")
print(encoder.classes_)

print("\nClass distribution:")

for emotion in encoder.classes_:
    print(f"{emotion}: {np.sum(y == emotion)}")


# ============================================================
# 6. SPLIT DATA BY ACTOR
# ============================================================

print("\nSplitting dataset by actors...")

# First split: 20% actors reserved for testing

splitter_1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE
)

train_val_idx, test_idx = next(
    splitter_1.split(
        X,
        y_encoded,
        groups=actor_ids
    )
)

# Second split: 20% of remaining actors for validation

splitter_2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE
)

train_relative_idx, val_relative_idx = next(
    splitter_2.split(
        X[train_val_idx],
        y_encoded[train_val_idx],
        groups=actor_ids[train_val_idx]
    )
)

# Convert relative indexes to original dataset indexes

train_idx = train_val_idx[train_relative_idx]
val_idx = train_val_idx[val_relative_idx]

X_train = X[train_idx]
y_train = y_encoded[train_idx]

X_val = X[val_idx]
y_val = y_encoded[val_idx]

X_test = X[test_idx]
y_test = y_encoded[test_idx]

print("\nData split completed!")

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))


# ============================================================
# 7. CHECK ACTOR SEPARATION
# ============================================================

train_actors = set(actor_ids[train_idx])
val_actors = set(actor_ids[val_idx])
test_actors = set(actor_ids[test_idx])

print("\nActor separation:")

print("Training actors:", sorted(train_actors))
print("Validation actors:", sorted(val_actors))
print("Testing actors:", sorted(test_actors))

print("\nActor overlap checks:")

print("Train-Val:", train_actors & val_actors)
print("Train-Test:", train_actors & test_actors)
print("Val-Test:", val_actors & test_actors)

if (
    train_actors & val_actors
    or train_actors & test_actors
    or val_actors & test_actors
):
    raise ValueError("Actor leakage detected!")


# ============================================================
# 8. CHECK CLASS DISTRIBUTION IN EACH SPLIT
# ============================================================

print("\nClass distribution in each split:")

for split_name, split_labels in [
    ("Training", y_train),
    ("Validation", y_val),
    ("Testing", y_test)
]:

    print(f"\n{split_name}:")

    for i, emotion in enumerate(encoder.classes_):
        count = np.sum(split_labels == i)
        print(f"{emotion}: {count}")


# ============================================================
# 9. NORMALIZATION
# ============================================================

print("\nPreparing normalization layer...")

normalizer = tf.keras.layers.Normalization(
    axis=None,
    name="spectrogram_normalization"
)

# Adapt only using training data to prevent data leakage

normalizer.adapt(X_train)

print("Normalization completed!")


# ============================================================
# 10. CLASS WEIGHTS
# ============================================================

print("\nCalculating class weights...")

classes_in_train = np.unique(y_train)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=classes_in_train,
    y=y_train
)

class_weights = {
    int(class_id): float(weight)
    for class_id, weight in zip(
        classes_in_train,
        class_weights_array
    )
}

print("Class weights:", class_weights)


# ============================================================
# 11. BUILD CNN MODEL
# ============================================================

input_shape = X_train.shape[1:]

print("\nCNN input shape:", input_shape)

model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=input_shape),

    # Normalization is included inside the saved model
    normalizer,

    # Convolution Block 1
    tf.keras.layers.Conv2D(
        32,
        kernel_size=(3, 3),
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    tf.keras.layers.Dropout(0.20),

    # Convolution Block 2
    tf.keras.layers.Conv2D(
        64,
        kernel_size=(3, 3),
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    tf.keras.layers.Dropout(0.25),

    # Convolution Block 3
    tf.keras.layers.Conv2D(
        128,
        kernel_size=(3, 3),
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    # Feature extraction
    tf.keras.layers.GlobalAveragePooling2D(),

    # Classification layers
    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.30),

    tf.keras.layers.Dense(
        num_classes,
        activation="softmax"
    )

])


# ============================================================
# 12. COMPILE MODEL
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)

model.summary()


# ============================================================
# 13. CALLBACKS
# ============================================================

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=0.000001,
        verbose=1
    ),

    tf.keras.callbacks.ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

]


# ============================================================
# 14. TRAIN CNN MODEL
# ============================================================

print("\n===================================")
print("STARTING CNN TRAINING")
print("===================================\n")

history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    class_weight=class_weights,

    callbacks=callbacks,

    shuffle=True,

    verbose=1

)


# ============================================================
# 15. LOAD BEST MODEL CHECKPOINT
# ============================================================

print("\nLoading best CNN checkpoint...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Best model checkpoint was not created!"
    )

# IMPORTANT:
# Load the best model selected using validation loss.
# This model also contains its fitted normalization layer.

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Best CNN checkpoint loaded successfully!")


# ============================================================
# 16. EVALUATE ON TEST DATA
# ============================================================

print("\n===================================")
print("EVALUATING MODEL")
print("===================================\n")

test_probabilities = model.predict(
    X_test,
    verbose=0
)

y_pred = np.argmax(
    test_probabilities,
    axis=1
)

test_loss = tf.keras.losses.sparse_categorical_crossentropy(
    y_test,
    test_probabilities
)

test_loss = float(tf.reduce_mean(test_loss).numpy())

test_accuracy = float(
    np.mean(y_pred == y_test)
)

print("Test Loss:", round(test_loss, 4))

print(
    "Test Accuracy:",
    round(test_accuracy * 100, 2),
    "%"
)

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        labels=np.arange(num_classes),
        target_names=encoder.classes_,
        zero_division=0
    )
)


# ============================================================
# 17. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(num_classes)
)

plt.figure(figsize=(10, 8))

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title("CNN Speech Emotion Recognition")

plt.colorbar()

tick_marks = np.arange(num_classes)

plt.xticks(
    tick_marks,
    encoder.classes_,
    rotation=45,
    ha="right"
)

plt.yticks(
    tick_marks,
    encoder.classes_
)

plt.xlabel("Predicted Emotion")
plt.ylabel("Actual Emotion")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=300,
    bbox_inches="tight"
)

print("\nConfusion matrix saved:", CONFUSION_MATRIX_PATH)


# ============================================================
# 18. TRAINING HISTORY
# ============================================================

plt.figure(figsize=(12, 5))

# Accuracy plot

plt.subplot(1, 2, 1)

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("CNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Loss plot

plt.subplot(1, 2, 2)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("CNN Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()

plt.savefig(
    HISTORY_PATH,
    dpi=300,
    bbox_inches="tight"
)

print("Training history saved:", HISTORY_PATH)


# ============================================================
# 19. SAVE FINAL MODEL AND LABELS
# ============================================================

# Save the reloaded best checkpoint, not the last training epoch.

model.save(MODEL_PATH)

np.save(
    LABELS_PATH,
    encoder.classes_
)

print("\n===================================")
print("CNN TRAINING COMPLETED SUCCESSFULLY")
print("===================================")

print("Model saved:", MODEL_PATH)
print("Labels saved:", LABELS_PATH)
print("Confusion matrix:", CONFUSION_MATRIX_PATH)
print("Training history:", HISTORY_PATH)

print("\nFinal test accuracy:", round(test_accuracy * 100, 2), "%")

# Display plots only after saving all files

plt.show()
