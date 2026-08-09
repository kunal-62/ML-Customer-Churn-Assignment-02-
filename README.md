# Machine Learning Model Evaluator Streamlit App

A Streamlit web application that allows users to upload a test dataset, select target labels, and evaluate pre-trained machine learning classification models (e.g., Logistic Regression) using metrics like **Accuracy**, **Precision**, **Recall**, and **ROC AUC**.

---

## 📁 Project Structure

```text
├── model/                     # Folder containing saved models and scalers
│   ├── log_reg_model.joblib   # Trained model file
│   └── scaler.joblib          # Fitted feature scaler
├── app.py                     # Main Streamlit application file
├── requirements.txt           # Python dependencies
├── .gitignore                 # Files and folders ignored by Git
└── README.md                  # Project documentation