import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, mean_absolute_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "saved_models"
VISUALIZATIONS_DIR = BASE_DIR / "visualizations"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 10, "figure.autolayout": True})


class _ANNNetClassifier(nn.Module):
    """Internal PyTorch Neural Network architecture for Multi-Class / Binary Classification."""
    def __init__(self, input_dim: int, hidden_layers: tuple, num_classes: int, activation: str = "relu", dropout_rate: float = 0.0):
        super().__init__()
        layers = []
        in_d = input_dim
        
        act_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "logistic": nn.Sigmoid,
            "sigmoid": nn.Sigmoid
        }
        act_fn = act_map.get(str(activation).lower(), nn.ReLU)

        for h in hidden_layers:
            layers.append(nn.Linear(in_d, h))
            layers.append(act_fn())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            in_d = h

        layers.append(nn.Linear(in_d, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class PyTorchANNClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-Learn-compatible PyTorch Artificial Neural Network (ANN) Classifier.
    """
    def __init__(
        self,
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        lr=0.001,
        max_iter=300,
        batch_size=32,
        weight_decay=0.0001,
        early_stopping=True,
        patience=15,
        random_state=42,
        dropout_rate=0.0
    ):
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.activation = activation
        self.lr = lr
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.early_stopping = early_stopping
        self.patience = patience
        self.random_state = random_state
        self.dropout_rate = dropout_rate
        self.classes_ = None
        self.model_ = None
        self.loss_curve_ = []

    def fit(self, X, y):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y)

        self.classes_ = np.unique(y_arr)
        num_classes = len(self.classes_)
        input_dim = X_arr.shape[1]

        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_indices = np.array([class_to_idx[val] for val in y_arr], dtype=np.int64)

        self.model_ = _ANNNetClassifier(
            input_dim=input_dim,
            hidden_layers=self.hidden_layer_sizes,
            num_classes=num_classes,
            activation=self.activation,
            dropout_rate=self.dropout_rate
        )

        dataset = TensorDataset(torch.from_numpy(X_arr), torch.from_numpy(y_indices))
        loader = DataLoader(dataset, batch_size=min(self.batch_size, len(X_arr)), shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.loss_curve_ = []
        best_loss = float("inf")
        patience_count = 0

        self.model_.train()
        for epoch in range(self.max_iter):
            epoch_loss = 0.0
            total_samples = 0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model_(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_x.size(0)
                total_samples += batch_x.size(0)

            avg_loss = epoch_loss / max(1, total_samples)
            self.loss_curve_.append(avg_loss)

            if self.early_stopping:
                if avg_loss < best_loss - 1e-4:
                    best_loss = avg_loss
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count >= self.patience:
                        break
        return self

    def predict_proba(self, X):
        if self.model_ is None:
            raise ValueError("Model has not been fitted yet.")
        self.model_.eval()
        X_arr = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            logits = self.model_(torch.from_numpy(X_arr))
            probs = torch.softmax(logits, dim=1).numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        pred_indices = np.argmax(probs, axis=1)
        return self.classes_[pred_indices]


class _ANNNetRegressor(nn.Module):
    """Internal PyTorch Neural Network architecture for Continuous Regression."""
    def __init__(self, input_dim: int, hidden_layers: tuple, activation: str = "relu", dropout_rate: float = 0.0):
        super().__init__()
        layers = []
        in_d = input_dim

        act_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "logistic": nn.Sigmoid,
            "sigmoid": nn.Sigmoid
        }
        act_fn = act_map.get(str(activation).lower(), nn.ReLU)

        for h in hidden_layers:
            layers.append(nn.Linear(in_d, h))
            layers.append(act_fn())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            in_d = h

        layers.append(nn.Linear(in_d, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class PyTorchANNRegressor(BaseEstimator, RegressorMixin):
    """
    Scikit-Learn-compatible PyTorch Artificial Neural Network (ANN) Regressor.
    """
    def __init__(
        self,
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        lr=0.005,
        max_iter=400,
        batch_size=32,
        weight_decay=0.0001,
        early_stopping=True,
        patience=20,
        random_state=42,
        dropout_rate=0.0
    ):
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.activation = activation
        self.lr = lr
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.early_stopping = early_stopping
        self.patience = patience
        self.random_state = random_state
        self.dropout_rate = dropout_rate
        self.model_ = None
        self.loss_curve_ = []

    def fit(self, X, y):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)

        input_dim = X_arr.shape[1]
        self.model_ = _ANNNetRegressor(
            input_dim=input_dim,
            hidden_layers=self.hidden_layer_sizes,
            activation=self.activation,
            dropout_rate=self.dropout_rate
        )

        dataset = TensorDataset(torch.from_numpy(X_arr), torch.from_numpy(y_arr))
        loader = DataLoader(dataset, batch_size=min(self.batch_size, len(X_arr)), shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.loss_curve_ = []
        best_loss = float("inf")
        patience_count = 0

        self.model_.train()
        for epoch in range(self.max_iter):
            epoch_loss = 0.0
            total_samples = 0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model_(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_x.size(0)
                total_samples += batch_x.size(0)

            avg_loss = epoch_loss / max(1, total_samples)
            self.loss_curve_.append(avg_loss)

            if self.early_stopping:
                if avg_loss < best_loss - 1e-4:
                    best_loss = avg_loss
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count >= self.patience:
                        break
        return self

    def predict(self, X):
        if self.model_ is None:
            raise ValueError("Model has not been fitted yet.")
        self.model_.eval()
        X_arr = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            preds = self.model_(torch.from_numpy(X_arr)).numpy()
        return preds


def build_preprocessor(X: pd.DataFrame):
    """
    Builds a robust scikit-learn ColumnTransformer fitted strictly on training data
    to eliminate data leakage.
    """
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
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues" if "Movie" in title else "Reds", xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title(f"{title} - Confusion Matrix", fontweight="bold")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    if loss_curve is not None and len(loss_curve) > 0:
        axes[1].plot(loss_curve, marker="o", markersize=3, color="#2b5c8f" if "Movie" in title else "#dc2626", linewidth=2, label="PyTorch CrossEntropy Loss")
        axes[1].set_title(f"{title} - ANN Loss Convergence", fontweight="bold")
        axes[1].set_xlabel("Epochs")
        axes[1].set_ylabel("CrossEntropy Loss")
        axes[1].legend()

    filepath = VISUALIZATIONS_DIR / filename
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close("all")
    print(f"[Plot Saved] -> {filepath}")
    return filepath


def plot_regression_results(y_true, y_pred, title, filename, loss_curve=None):
    fig, axes = plt.subplots(1, 2 if loss_curve is not None else 1, figsize=(12, 5))
    if loss_curve is None:
        axes = [axes]

    axes[0].scatter(y_true, y_pred, alpha=0.7, color="#2b5c8f", edgecolors="w", s=40)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    axes[0].plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Fit (y=x)")
    axes[0].set_title(f"{title} - Actual vs. Predicted", fontweight="bold")
    axes[0].set_xlabel("Actual Values")
    axes[0].set_ylabel("Predicted Values")
    axes[0].legend()

    if loss_curve is not None and len(loss_curve) > 0:
        axes[1].plot(loss_curve, marker="o", markersize=3, color="#27ae60", linewidth=2, label="PyTorch MSE Loss")
        axes[1].set_title(f"{title} - ANN Loss Convergence", fontweight="bold")
        axes[1].set_xlabel("Epochs")
        axes[1].set_ylabel("MSE Loss")
        axes[1].legend()

    filepath = VISUALIZATIONS_DIR / filename
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close("all")
    print(f"[Plot Saved] -> {filepath}")
    return filepath
