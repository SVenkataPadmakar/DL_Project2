# 🫁 Lung Cancer Prediction (Artificial Neural Network - ANN)

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-v2.0+-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.3+-orange.svg)](https://scikit-learn.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Interactive%20UI-FF4B4B.svg)](https://streamlit.io/)
[![Pytest](https://img.shields.io/badge/pytest-Passing-brightgreen.svg)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Repository](https://img.shields.io/badge/GitHub-SVenkataPadmakar%2FDL__Project2-181717.svg?logo=github)](https://github.com/SVenkataPadmakar/DL_Project2)

An end-to-end, production-grade clinical **Artificial Neural Network (ANN)** deep learning project built in **PyTorch** for binary lung cancer diagnosis and risk prediction based on patient demographics, lifestyle habits, and clinical indicators.

---

## 🌟 Highlights & Features
- **Deep ANN Architecture**: Multi-Layer PyTorch Neural Network (`Dense(128) -> Dense(64) -> Dense(32) -> Binary Output`) with ReLU activations, Adam optimization, and CrossEntropyLoss backpropagation.
- **Leakage-Free Preprocessing**: `ColumnTransformer` with `StandardScaler` for continuous numeric features and `OneHotEncoder` for categorical factors, fitted strictly on training data.
- **Diagnostics & Visualizations**: Automatic confusion matrix heatmaps and PyTorch epoch loss convergence curves saved to [`visualizations/`](visualizations/).
- **Model Serialization**: Trained PyTorch ANN model artifact and preprocessing pipeline saved to [`saved_models/lung_cancer_model.pkl`](saved_models/lung_cancer_model.pkl).
- **Interactive Web Studio**: Streamlit application (`app.py`) for clinical data exploration, interactive ANN training with hyperparameter tuning, and real-time live patient risk assessment.
- **Automated Tests**: Pytest suite in [`tests/`](tests/) validating data schema, PyTorch ANN training, and predictions.

---

## 📁 Repository Structure

```
DL_Project2/
├── lung_cancer.csv                   # Dataset
├── train.py                          # PyTorch ANN training pipeline
├── app.py                            # Interactive Streamlit Web Studio
├── model_utils.py                    # PyTorch ANN classifier, preprocessing & plotting
├── requirements.txt                  # Python dependencies (includes torch)
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

### 3. Train the PyTorch Artificial Neural Network (ANN)
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
