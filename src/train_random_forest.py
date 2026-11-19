
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Load extracted features
data = pd.read_csv("data/processed/features.csv")

X = data.drop(columns=["filename", "actor_id", "emotion"])
y = data["emotion"]
groups = data["actor_id"]

# Use the same actor-based split as our SVM experiment
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(splitter.split(X, y, groups))

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest...")
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\n===== RANDOM FOREST RESULTS =====")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
labels = sorted(y.unique())

cm = confusion_matrix(y_test, y_pred, labels=labels)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

disp.plot(cmap="Greens", xticks_rotation=45)

plt.title("Speech Emotion Recognition - Random Forest")
plt.tight_layout()

os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/random_forest_confusion_matrix.png", dpi=150)

# Save model
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/random_forest_model.joblib")

print("\nRandom Forest model saved successfully!")
print("Location: models/random_forest_model.joblib")

plt.show()
