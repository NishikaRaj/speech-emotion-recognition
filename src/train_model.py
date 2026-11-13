import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# 1. Load dataset
df = pd.read_csv("data/processed/features.csv")

print("Dataset loaded:", df.shape)

# 2. Separate features and labels
X = df.drop(columns=["filename", "actor_id", "emotion"])
y = df["emotion"]

# 3. Split by actor to prevent actor leakage
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=df["actor_id"])
)

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# 4. Build ML pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        class_weight="balanced"
    ))
])

# 5. Train model
print("\nTraining SVM model...")
model.fit(X_train, y_train)

# 6. Make predictions
y_pred = model.predict(X_test)

# 7. Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Confusion Matrix
labels = sorted(y.unique())

cm = confusion_matrix(y_test, y_pred, labels=labels)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)


disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Speech Emotion Recognition - SVM")
plt.tight_layout()

# Save confusion matrix
plt.savefig("outputs/confusion_matrix.png", dpi=150)

# Save trained model BEFORE displaying the graph
import os

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/svm_emotion_model.joblib")

print("\nModel saved successfully!")
print("Model exists:", os.path.exists("models/svm_emotion_model.joblib"))

# Display confusion matrix
plt.show()
