import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from model_utils import build_preprocessor, save_model, plot_classification_results

DATASET_PATH = Path(__file__).resolve().parent / "lung_cancer.csv"


def train_model(csv_path=DATASET_PATH):
    print("=" * 60)
    print(" [+] Project 2: Training Deep Neural Network for Lung Cancer Prediction")
    print("=" * 60)
    
    df = pd.read_csv(csv_path)
    target = "Lung_Cancer"
    
    X = df.drop(columns=[target])
    y_raw = df[target]
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw.astype(str))
    class_names = [str(c) for c in label_encoder.classes_]
    
    # Stratified Train/Test split without data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    preprocessor, num_cols, cat_cols = build_preprocessor(X_train)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    # Deep Neural Network: (128 -> 64 -> 32)
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=0.001,
        batch_size=32,
        learning_rate_init=0.001,
        max_iter=300,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=15,
        verbose=False
    )
    
    model.fit(X_train_proc, y_train)
    
    y_pred = model.predict(X_test_proc)
    y_prob = model.predict_proba(X_test_proc)[:, 1] if len(class_names) == 2 else model.predict_proba(X_test_proc)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {acc:.4f} ({acc * 100:.2f}%)\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    if len(class_names) == 2:
        try:
            auc = roc_auc_score(y_test, y_prob)
            print(f"ROC-AUC Score: {auc:.4f}")
        except Exception:
            pass
            
    # Save artifacts
    plot_classification_results(
        y_test, y_pred, y_prob, class_names,
        title="Lung Cancer Prediction",
        filename="lung_cancer_evaluation.png",
        loss_curve=model.loss_curve_
    )
    
    save_model({
        "model": model,
        "preprocessor": preprocessor,
        "label_encoder": label_encoder,
        "class_names": class_names,
        "accuracy": acc
    }, "lung_cancer_model.pkl")
    
    return acc, model


if __name__ == "__main__":
    train_model()
