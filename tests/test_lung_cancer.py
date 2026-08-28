import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import pandas as pd
from train import train_model
from model_utils import load_model, VISUALIZATIONS_DIR


def test_lung_cancer_dataset_exists():
    p = PROJECT_DIR / "lung_cancer.csv"
    assert p.exists()
    df = pd.read_csv(p)
    assert not df.empty
    assert "Lung_Cancer" in df.columns


def test_lung_cancer_training_and_artifacts():
    acc, model = train_model()
    assert 0.0 <= acc <= 1.0
    saved = load_model("lung_cancer_model.pkl")
    assert "model" in saved
    assert "preprocessor" in saved
    assert (VISUALIZATIONS_DIR / "lung_cancer_evaluation.png").exists()
