
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

df = pd.read_csv("data/processed/features.csv")

y = df["emotion"]
groups = df["actor_id"]

mfcc = df.filter(regex=r"^mfcc_")

mfcc_rms_zcr = df.drop(columns=[
    "filename", "actor_id", "emotion",
    "pitch_mean", "pitch_std", "pitch_min", "pitch_max"
])

all_features = df.drop(columns=[
    "filename", "actor_id", "emotion"
])

feature_sets = {
    "MFCC only": mfcc,
    "MFCC + RMS + ZCR": mfcc_rms_zcr,
    "MFCC + RMS + ZCR + Pitch": all_features
}

cv = GroupKFold(n_splits=5)

scoring = {
    "accuracy": "accuracy",
    "macro_f1": "f1_macro",
    "balanced_accuracy": "balanced_accuracy"
}

for name, X in feature_sets.items():
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            class_weight="balanced"
        ))
    ])

    results = cross_validate(
        model, X, y,
        cv=cv,
        groups=groups,
        scoring=scoring
    )

    print("\n" + name)

    for metric in scoring:
        scores = results["test_" + metric] * 100
        print(
            metric,
            "Mean:", round(scores.mean(), 2),
            "Std:", round(scores.std(), 2)
        )
