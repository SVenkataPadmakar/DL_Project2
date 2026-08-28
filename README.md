# 🫁 Lung Cancer Risk Prediction (Deep Learning)

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.3+-orange.svg)](https://scikit-learn.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Interactive%20UI-FF4B4B.svg)](https://streamlit.io/)
[![Pytest](https://img.shields.io/badge/pytest-Passing-brightgreen.svg)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Repository](https://img.shields.io/badge/GitHub-SVenkataPadmakar%2FDL__Project2-181717.svg?logo=github)](https://github.com/SVenkataPadmakar/DL_Project2)

An end-to-end, production-grade **Deep Neural Network (DNN)** project for predicting patient lung cancer risk based on demographic, lifestyle, and clinical factors.

---

## 🌟 Highlights & Features
- **Deep Architecture**: Multi-Layer Perceptron (`128 -> 64 -> 32 -> Sigmoid`) with ReLU non-linearities and Adam optimizer.
- **Leakage-Free Preprocessing**: `ColumnTransformer` with `StandardScaler` for continuous features (`Age`, `Anxiety`, `Peer_Pressure`) and `OneHotEncoder` for categorical indicators (`Smoking`, `Chronic_Disease`), fitted strictly on training data.
- **Diagnostics & Visualizations**: Automatic confusion matrix heatmaps and loss convergence curves saved to [`visualizations/`](visualizations/).
- **Model Serialization**: Trained model and preprocessing pipeline saved to [`saved_models/lung_cancer_model.pkl`](saved_models/lung_cancer_model.pkl).
- **Clinical Web Studio**: Interactive Streamlit application (`app.py`) for live risk assessment and feature impact analysis.
- **Unit & Integration Tests**: Pytest test suite in [`tests/`](tests/) validating data schema, model training, and inference.

---

## 📁 Repository Structure

```
DL_Project2/
├── lung_cancer.csv                   # Patient dataset
├── train.py                          # Deep Learning training pipeline
├── app.py                            # Interactive Streamlit Web Studio
├── model_utils.py                    # Preprocessing, plotting, and model persistence
├── requirements.txt                  # Python dependencies
├── README.md                         # Documentation
├── LICENSE                           # MIT License
├── .gitignore                        # Git ignore rules
├── saved_models/                     # Serialized model artifacts
│   └── lung_cancer_model.pkl
├── visualizations/                   # Evaluation charts
│   └── lung_cancer_evaluation.png
└── tests/                            # Automated test suite
    └── test_lung_cancer.py
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/SVenkataPadmakar/DL_Project2.git
cd DL_Project2
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Deep Neural Network
```bash
python train.py
```

### 4. Launch the Interactive Web Application
```bash
streamlit run app.py
```

### 5. Run Automated Tests
```bash
pytest -v
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
