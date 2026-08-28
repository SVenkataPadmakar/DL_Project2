import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "saved_models"
VISUALIZATIONS_DIR = BASE_DIR / "visualizations"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 10, "figure.autolayout": True})


def build_preprocessor(X: pd.DataFrame):
    numeric_features = X.select_dtypes(include=["int64", "float64", "number"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    return preprocessor, numeric_features, categorical_features


def save_model(model_artifact, filename: str):
    filepath = MODELS_DIR / filename
    joblib.dump(model_artifact, filepath)
    print(f"[Model Saved] -> {filepath}")
    return filepath


def load_model(filename: str):
    filepath = MODELS_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    return joblib.load(filepath)


def plot_classification_results(y_true, y_pred, y_prob, class_names, title, filename, loss_curve=None):
    fig, axes = plt.subplots(1, 2 if loss_curve is not None else 1, figsize=(12, 5))
    if loss_curve is None:
        axes = [axes]

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title(f"{title} - Confusion Matrix", fontweight="bold")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    if loss_curve is not None and len(loss_curve) > 0:
        axes[1].plot(loss_curve, marker="o", color="#dc2626", linewidth=2, label="Training Loss")
        axes[1].set_title(f"{title} - Loss Convergence", fontweight="bold")
        axes[1].set_xlabel("Iterations / Epochs")
        axes[1].set_ylabel("Loss")
        axes[1].legend()

    filepath = VISUALIZATIONS_DIR / filename
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close("all")
    print(f"[Plot Saved] -> {filepath}")
    return filepath
