import joblib
import os
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score, 
    matthews_corrcoef
)


#1. SET UP THE PAGE
st.set_page_config(page_title="Customer Churn Prediction App", layout="centered")

st.title("Classification Model Evaluator")

#2. SET UP MODEL SELECTION OPTIONS
model_options = ["Logistic Regression", "Decision Tree", "kNN", "Naives Bayes", "Random Forest"]
selection = st.sidebar.selectbox('Select the classification model you would like to evaluate ?', model_options)
st.subheader(selection)

#3. Load trained model and scaler
#@st.cache_resource
def load_artifacts():
    scaler = joblib.load("models/scaler.joblib")

    models = {"Logistic Regression" :"log_reg_model.joblib",
              "Decision Tree": "dt_model.joblib",
              "kNN": "kNN_model.joblib",
              "Naives Bayes": "nb_model_tuned.joblib",
              "Random Forest": "rf_model.joblib",
            }

    model_path = os.path.join("models", models[selection])
    model = joblib.load(model_path)
    return scaler, model

try:
    scaler, model = load_artifacts()
    st.success("Model and scaler loaded successfully!")
except Exception as e:
    st.error(f"Error loading model/scaler files: {e}")
    st.stop()

#4. Upload test dataset
uploaded_file = st.file_uploader("Upload your test dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", df_test.head())

    # 3. Select target column
    target_col = st.selectbox("Select the Target (True Label) Column", df_test.columns)

    if st.button("Evaluate Model"):
        # Separate features and target
        X_test = df_test.drop(columns=[target_col])
        y_test = df_test[target_col]

        # Preprocess features using saved scaler
        X_test_scaled = scaler.transform(X_test)

        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        # # Calculate metrics
        # acc = accuracy_score(y_test, y_pred)
        # prec = precision_score(y_test, y_pred)
        # rec = recall_score(y_test, y_pred)
        # roc_auc = roc_auc_score(y_test, y_prob)

        # # 4. Display metrics in columns
        # st.subheader("Model Performance Metrics")
        # col1, col2, col3, col4 = st.columns(4)
        # col1.metric("Accuracy", f"{acc:.4f}")
        # col2.metric("Precision", f"{prec:.4f}")
        # col3.metric("Recall", f"{rec:.4f}")
        # col4.metric("ROC AUC", f"{roc_auc:.4f}")

        # # 5. Additional evaluation details
        # st.subheader("Confusion Matrix")
        # st.dataframe(pd.DataFrame(confusion_matrix(y_test, y_pred)))

        # st.subheader("Classification Report")
        # st.text(classification_report(y_test, y_pred))

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        # Display metrics in columns
        st.subheader("📊 Model Performance Metrics")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("Precision", f"{prec:.4f}")
        col3.metric("Recall", f"{rec:.4f}")
        col4.metric("F1 Score", f"{f1:.4f}")
        col5.metric("MCC", f"{mcc:.4f}")
        col6.metric("ROC AUC", f"{roc_auc:.4f}")

        # Confusion Matrix with heatmap
        st.subheader("🔎 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # Classification Report in a styled dataframe
        st.subheader("📑 Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.background_gradient(cmap="YlGnBu"))