from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

from model_utils import build_preprocessor, save_model, load_model, PyTorchANNClassifier

st.set_page_config(
    page_title="Lung Cancer Prediction ANN Studio",
    page_icon="🫁",
    layout="wide"
)

st.title("🫁 Lung Cancer Artificial Neural Network (ANN) Risk Assessment")
st.markdown("Binary **PyTorch Artificial Neural Network (ANN)** predicting the probability of lung cancer from patient age, smoking habits, anxiety, peer pressure, and chronic disease.")

DATA_FILE = Path(__file__).resolve().parent / "lung_cancer.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)

df = load_data()

tab1, tab2, tab3 = st.tabs(["📊 Patient Data Profile", "⚡ PyTorch ANN Studio", "🔮 Live Risk Predictor"])

# TAB 1
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Patients", len(df))
    with col2:
        st.metric("Risk Features", len(df.columns) - 1)
    with col3:
        st.metric("Cancer Positives", (df["Lung_Cancer"] == "Yes").sum())
    with col4:
        st.metric("Cancer Negatives", (df["Lung_Cancer"] == "No").sum())

    st.subheader("Patient Records Preview")
    st.dataframe(df.head(10), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Cancer Status Breakdown")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x="Lung_Cancer", palette="Reds", ax=ax)
        st.pyplot(fig)
    with col_b:
        st.subheader("Age Distribution vs. Cancer")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x="Lung_Cancer", y="Age", palette="Set2", ax=ax2)
        st.pyplot(fig2)

# TAB 2
with tab2:
    st.subheader("Interactive PyTorch ANN Model Training")
    c1, c2, c3 = st.columns(3)
    with c1:
        l1 = st.slider("Layer 1 Neurons", 32, 256, 128, step=16)
        l2 = st.slider("Layer 2 Neurons", 16, 128, 64, step=16)
        l3 = st.slider("Layer 3 Neurons", 0, 64, 32, step=8)
    with c2:
        activation = st.selectbox("Activation Function", ["relu", "tanh", "sigmoid"])
        lr = st.select_slider("Learning Rate", options=[0.0005, 0.001, 0.005, 0.01], value=0.001)
        epochs = st.slider("Epochs", 50, 500, 200, step=25)
    with c3:
        test_size = st.slider("Validation Split", 0.1, 0.4, 0.2, step=0.05)
        early_stop = st.checkbox("Early Stopping", value=True)

    layers = [l1, l2] if l3 == 0 else [l1, l2, l3]

    if st.button("🚀 Train PyTorch ANN", type="primary"):
        with st.spinner("Training PyTorch Artificial Neural Network..."):
            X = df.drop(columns=["Lung_Cancer"])
            y_raw = df["Lung_Cancer"]
            le = LabelEncoder()
            y = le.fit_transform(y_raw)
            class_names = list(le.classes_)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

            preprocessor, _, _ = build_preprocessor(X_train)
            X_train_proc = preprocessor.fit_transform(X_train)
            X_test_proc = preprocessor.transform(X_test)

            model = PyTorchANNClassifier(
                hidden_layer_sizes=tuple(layers),
                activation=activation,
                lr=lr,
                max_iter=epochs,
                early_stopping=early_stop,
                random_state=42
            )
            model.fit(X_train_proc, y_train)
            y_pred = model.predict(X_test_proc)
            acc = accuracy_score(y_test, y_pred)

            st.success(f"Training Complete! Test Accuracy: {acc*100:.2f}%")

            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.subheader("Confusion Matrix")
                fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", xticklabels=class_names, yticklabels=class_names, ax=ax_cm)
                st.pyplot(fig_cm)
            with col_res2:
                st.subheader("PyTorch Loss Convergence")
                fig_l, ax_l = plt.subplots(figsize=(6, 4))
                ax_l.plot(model.loss_curve_, color="#dc2626", lw=2, marker="o", markersize=3)
                ax_l.set_xlabel("Epochs")
                ax_l.set_ylabel("CrossEntropy Loss")
                st.pyplot(fig_l)

# TAB 3
with tab3:
    st.subheader("Patient Live Assessment (PyTorch ANN)")
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        in_age = st.slider("Patient Age", min_value=18, max_value=90, value=55)
        in_smoking = st.selectbox("Smoking Habit", ["Yes", "No"])
        in_chronic = st.selectbox("Chronic Disease History", ["Yes", "No"])
    with col_in2:
        in_anxiety = st.slider("Anxiety Level (1-10)", min_value=1, max_value=10, value=5)
        in_peer = st.slider("Peer Pressure (1-10)", min_value=1, max_value=10, value=5)

    if st.button("🔮 Evaluate Risk", type="primary"):
        sample_df = pd.DataFrame([{
            "Age": in_age,
            "Smoking": in_smoking,
            "Anxiety": in_anxiety,
            "Peer_Pressure": in_peer,
            "Chronic_Disease": in_chronic
        }])

        X = df.drop(columns=["Lung_Cancer"])
        y_raw = df["Lung_Cancer"]
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        class_names = list(le.classes_)

        preprocessor, _, _ = build_preprocessor(X)
        X_proc = preprocessor.fit_transform(X)
        model = PyTorchANNClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=200, random_state=42)
        model.fit(X_proc, y)

        sample_proc = preprocessor.transform(sample_df)
        pred_label = model.predict(sample_proc)[0]
        pred_cancer = pred_label if isinstance(pred_label, str) else class_names[pred_label]
        probs = model.predict_proba(sample_proc)[0]

        if pred_cancer == "Yes" or pred_cancer == 1:
            st.error(f"⚠️ **High Risk Assessment:** Probability of Lung Cancer is **{probs[1]*100:.1f}%**")
        else:
            st.success(f"✅ **Low Risk Assessment:** Probability of Lung Cancer is **{probs[0]*100:.1f}%**")

        prob_df = pd.DataFrame({"Outcome": class_names, "Probability": probs})
        fig_p, ax_p = plt.subplots(figsize=(6, 3))
        sns.barplot(data=prob_df, x="Probability", y="Outcome", palette="Reds", ax=ax_p)
        ax_p.set_xlim(0, 1.0)
        st.pyplot(fig_p)
