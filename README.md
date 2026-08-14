# Telco Customer Churn Prediction

## 1. Project Overview

This project predicts customer churn for a telecommunications company
using supervised machine learning classification algorithms.

The objective is to identify customers likely to churn so that the
business can take proactive retention actions. Because churn prediction
is an imbalanced classification problem, the project evaluates multiple
metrics rather than relying only on Accuracy.

The main models evaluated are:

-   Logistic Regression
-   Decision Tree
-   k-Nearest Neighbors (kNN)
-   Gaussian Naive Bayes
-   Random Forest

------------------------------------------------------------------------

## 2. Dataset

The project uses the Telco Customer Churn dataset.

The processed dataset contains demographic, account, service, contract,
payment, and billing features.

Examples include:

-   `gender`
-   `SeniorCitizen`
-   `Partner`
-   `Dependents`
-   `tenure`
-   `PhoneService`
-   `PaperlessBilling`
-   `MonthlyCharges`
-   `TotalCharges`
-   `MultipleLines`
-   `InternetService`
-   `OnlineSecurity`
-   `OnlineBackup`
-   `DeviceProtection`
-   `TechSupport`
-   `StreamingTV`
-   `StreamingMovies`
-   `Contract`
-   `PaymentMethod`

### Target

`Churn`

-   `0` = Customer did not churn
-   `1` = Customer churned

Categorical variables were encoded into numerical features before model
training.

------------------------------------------------------------------------

## 3. Project Workflow

``` text
Raw Dataset
     |
     v
Data Preprocessing
     |
     v
Categorical Encoding
     |
     v
Train / Test Split
     |
     +-----------------------------+
     |                             |
     v                             v
Feature Scaling               Tree-Based Models
     |                             |
     |                    Decision Tree / Random Forest
     |                             |
     +-------------+---------------+
                   |
                   v
            Baseline Models
                   |
                   v
           Hyperparameter Tuning
              GridSearchCV
                   |
                   v
             Tuned Models
                   |
                   v
          Test Set Evaluation
                   |
                   v
        Model Comparison & Selection
```

------------------------------------------------------------------------

## 4. Feature Scaling

  Model                  Scaling
  ---------------------- ---------
  Logistic Regression    Yes
  Decision Tree          No
  kNN                    Yes
  Gaussian Naive Bayes   Yes
  Random Forest          No

For models requiring scaling, `StandardScaler` was fitted only on the
training data:

``` python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The fitted scaler can be saved with Joblib:

``` python
import joblib

joblib.dump(scaler, "saved_data/scaler.joblib")
```

------------------------------------------------------------------------

## 5. Hyperparameter Tuning

`GridSearchCV` with 5-fold cross-validation was used. The main
optimization metric was F1 Score:

``` python
scoring="f1"
```

### Best configurations

**Logistic Regression**

``` text
C = 1
class_weight = balanced
solver = liblinear
CV F1 = 0.6325
```

**Decision Tree**

``` text
class_weight = balanced
criterion = gini
max_depth = 7
min_samples_leaf = 20
min_samples_split = 2
CV F1 = 0.6213
```

**kNN**

``` text
metric = manhattan
n_neighbors = 15
weights = uniform
CV F1 = 0.5778
```

**Naive Bayes**

``` text
var_smoothing = 1e-11
CV F1 = 0.6235
```

**Random Forest**

``` text
class_weight = balanced
max_depth = 10
max_features = sqrt
min_samples_leaf = 5
n_estimators = 100
CV F1 = 0.6338
```

------------------------------------------------------------------------

## 6. Baseline Results

  Model                   Accuracy   Precision   Recall   ROC-AUC
  --------------------- ---------- ----------- -------- ---------
  Logistic Regression       0.8038      0.6476   0.5749    0.8357
  Decision Tree             0.7015      0.4401   0.4519    0.6218
  kNN                       0.7548      0.5387   0.5401    0.7682
  Naive Bayes               0.7356      0.5018   0.7513    0.8219
  Random Forest             0.7889      0.6262   0.5107    0.8209

------------------------------------------------------------------------

## 7. Tuned Model Results

  --------------------------------------------------------------------------------------------
  ML Model           Accuracy          AUC    Precision       Recall           F1          MCC
  -------------- ------------ ------------ ------------ ------------ ------------ ------------
  **Logistic           0.7264   **0.8349**       0.4909       0.7968   **0.6075**   **0.4439**
  Regression**                                                                    

  Decision Tree        0.5800       0.7859       0.3835   **0.9545**       0.5471       0.3724

  **kNN**          **0.7747**       0.8133   **0.5777**       0.5668       0.5722       0.4193

  Naive Bayes          0.7356       0.8219       0.5018       0.7513       0.6017       0.4343

  Random Forest        0.5800       0.8121       0.3814   **0.9332**       0.5415       0.3569
  --------------------------------------------------------------------------------------------

------------------------------------------------------------------------

## 8. Metric Definitions

### Accuracy

Percentage of all predictions that are correct.

### Precision

Of the customers predicted to churn, the proportion that actually
churned.

### Recall

Of the customers who actually churned, the proportion correctly
identified.

Recall is especially important when missing a potential churner is
costly.

### F1 Score

The harmonic mean of Precision and Recall:

``` text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### ROC-AUC

Measures how well a model distinguishes churners from non-churners
across classification thresholds.

### MCC

Matthews Correlation Coefficient considers TP, TN, FP, and FN and is
useful for imbalanced classification.

------------------------------------------------------------------------

# 9. Observations About Model Performance

## Logistic Regression

Logistic Regression provides the strongest overall balance.

After tuning:

-   Accuracy: **72.64%**
-   Precision: **49.09%**
-   Recall: **79.68%**
-   F1: **60.75%**
-   ROC-AUC: **83.49%**
-   MCC: **0.4439**
-   CV F1: **0.6325**

The `class_weight="balanced"` setting increased Recall from **57.49% to
79.68%** compared with the baseline. This reduced missed churners but
also increased false positives.

It has the highest tuned ROC-AUC and MCC, indicating strong
discrimination and the strongest overall balanced classification
performance.

**Observation:** Logistic Regression is the strongest overall model for
this dataset.

------------------------------------------------------------------------

## Decision Tree

The tuned Decision Tree achieved:

-   Accuracy: **58.00%**
-   Precision: **38.35%**
-   Recall: **95.45%**
-   F1: **54.71%**
-   ROC-AUC: **78.59%**
-   MCC: **0.3724**
-   CV F1: **62.13%**

It achieved the highest Recall of all models, identifying **95.45% of
actual churners**.

However, its low Precision and Accuracy indicate many false-positive
churn predictions.

The selected `max_depth=7` and `min_samples_leaf=20` provide useful
control over tree complexity.

**Observation:** Decision Tree is appropriate when minimizing missed
churners is the highest priority, but it is weaker in overall balanced
performance.

------------------------------------------------------------------------

## kNN

The tuned kNN model achieved:

-   Accuracy: **77.47%**
-   Precision: **57.77%**
-   Recall: **56.68%**
-   F1: **57.22%**
-   ROC-AUC: **81.33%**
-   MCC: **0.4193**
-   CV F1: **57.78%**

kNN achieved the **highest test Accuracy and Precision** among the tuned
models.

However, its Recall of **56.68%** means that it misses a relatively
large proportion of actual churners.

The best configuration was:

``` text
metric = manhattan
n_neighbors = 15
weights = uniform
```

**Observation:** kNN is good at overall classification accuracy and
precision but is less effective when the goal is to identify as many
churners as possible.

------------------------------------------------------------------------

## Gaussian Naive Bayes

The tuned Naive Bayes model achieved:

-   Accuracy: **73.56%**
-   Precision: **50.18%**
-   Recall: **75.13%**
-   F1: **60.17%**
-   ROC-AUC: **82.19%**
-   MCC: **0.4343**
-   CV F1: **62.35%**

Naive Bayes provides a good balance between Recall and Precision.

Its Recall of **75.13%** and ROC-AUC of **82.19%** make it a competitive
alternative to Logistic Regression.

Although Naive Bayes assumes conditional independence between features,
several Telco features are naturally related. Despite this limitation,
it performed well.

**Observation:** Naive Bayes is a strong, simple baseline and provides
competitive churn detection performance.

------------------------------------------------------------------------

## Random Forest

The tuned Random Forest achieved:

-   Accuracy: **58.00%**
-   Precision: **38.14%**
-   Recall: **93.32%**
-   F1: **54.15%**
-   ROC-AUC: **81.21%**
-   MCC: **0.3569**
-   CV F1: **63.38%**

Random Forest achieved the **highest CV F1 score (0.6338)**, narrowly
ahead of Logistic Regression (0.6325).

It also achieved very high Recall of **93.32%**.

However, its test-set Precision and Accuracy were relatively low,
indicating a large number of false positives.

**Observation:** Random Forest is highly effective at detecting
potential churners but is less balanced on the test set than Logistic
Regression.

------------------------------------------------------------------------

# 10. Metric-Wise Best Models

  Metric       Best Model                       Score
  ------------ ------------------------- ------------
  Accuracy     **kNN**                     **0.7747**
  ROC-AUC      **Logistic Regression**     **0.8349**
  Precision    **kNN**                     **0.5777**
  Recall       **Decision Tree**           **0.9545**
  F1           **Logistic Regression**     **0.6075**
  MCC          **Logistic Regression**     **0.4439**
  Best CV F1   **Random Forest**           **0.6338**

No single model is best on every metric.

------------------------------------------------------------------------

# 11. Overall Winner

## Logistic Regression

**Logistic Regression is selected as the overall winner** for this
project.

The main reasons are:

-   Highest test ROC-AUC: **0.8349**
-   Highest test MCC: **0.4439**
-   Highest test F1: **0.6075**
-   Strong Recall: **79.68%**
-   Best overall balance between Precision and Recall
-   CV F1 of **0.6325**, very close to the highest CV F1 of Random
    Forest (**0.6338**)

The difference between Random Forest and Logistic Regression in CV F1 is
only:

``` text
0.6338 - 0.6325 = 0.0013
```

Therefore, Random Forest's very small CV advantage does not outweigh its
substantially weaker test-set Precision, Accuracy, F1, and MCC.

### Business Perspective

If the business's primary objective is to **identify almost every
customer who may churn**, Decision Tree or Random Forest may be
preferred because their Recall is above 93%.

However, these models generate many false positives, potentially causing
the company to spend retention resources on customers who were not
actually going to churn.

Logistic Regression offers a more balanced approach: it identifies a
large proportion of churners while maintaining stronger overall
predictive quality.

------------------------------------------------------------------------

# 12. Key Findings

1.  Hyperparameter tuning substantially changed the behavior of several
    models, especially models using `class_weight="balanced"`.
2.  Decision Tree achieved the highest Recall at **95.45%**.
3.  Random Forest achieved the highest cross-validation F1 at
    **0.6338**.
4.  kNN achieved the highest test Accuracy at **77.47%** and Precision
    at **57.77%**.
5.  Logistic Regression achieved the highest test ROC-AUC at **83.49%**.
6.  Logistic Regression achieved the highest test F1 and MCC.
7.  Naive Bayes provided a competitive balance of Recall, F1, and
    ROC-AUC.
8.  Accuracy alone would not be sufficient for selecting the best churn
    model.
9.  The class imbalance makes Recall and F1 particularly important.
10. Logistic Regression was selected as the overall model because it
    provides the strongest balance across the evaluation metrics.

------------------------------------------------------------------------

# 13. Model Saving

Models and preprocessing artifacts can be saved using Joblib:

``` python
import joblib

joblib.dump(model, "model.joblib")
```

Load a saved model:

``` python
model = joblib.load("model.joblib")
```

Save the fitted scaler:

``` python
joblib.dump(scaler, "scaler.joblib")
```

------------------------------------------------------------------------

# 14. Project Structure

```text
Telco_Customer_Churn_ML_Asgn02/
│
├── dataset/
│   └── Telco_Customer_Churn.csv
│
├── models/
│   ├── saved_data/
│   │   ├── X_train.csv
│   │   └── y_train.csv
│   │
│   ├── DecisionTree.ipynb
│   ├── dt_model.joblib
│   ├── kNN_model.joblib
│   ├── kNN.ipynb
│   ├── log_reg_model.joblib
│   ├── LogisticRegression.ipynb
│   ├── ml_model_creation.ipynb
│   ├── NaiveBayes.ipynb
│   ├── nb_model_tuned.joblib
│   ├── RandomForest.ipynb
│   ├── rf_model.joblib
│   ├── scaler.joblib
│   └── svm_model.joblib
│
├── venv/
│
├── .gitignore
├── app.py
├── README.md
├── requirements.txt
└── test_data.csv

------------------------------------------------------------------------

# 15. Technologies Used

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   Joblib
-   Jupyter Notebook

### Machine Learning Techniques

-   Logistic Regression
-   Decision Tree Classification
-   k-Nearest Neighbors
-   Gaussian Naive Bayes
-   Random Forest Ensemble
-   GridSearchCV
-   5-Fold Cross-Validation
-   StandardScaler

------------------------------------------------------------------------

# 16. Future Improvements

Potential improvements include:

-   Optimize classification probability thresholds according to the
    business cost of false positives and false negatives.
-   Compare Precision-Recall curves.
-   Perform additional feature engineering.
-   Analyze Logistic Regression coefficients.
-   Analyze Decision Tree and Random Forest feature importance.
-   Use SHAP or another explainability technique.
-   Build a Scikit-learn Pipeline combining preprocessing and the
    selected model.
-   Deploy the selected model through an API or web application.
-   Introduce cost-sensitive evaluation based on actual customer
    retention costs.

------------------------------------------------------------------------

# 17. Conclusion

This project demonstrates a complete machine learning workflow for Telco
Customer Churn prediction, including preprocessing, baseline modeling,
hyperparameter tuning, cross-validation, test-set evaluation, and model
comparison.

The experiments show that model selection should not be based on
Accuracy alone. Different models optimize different aspects of the
problem:

-   Decision Tree is strongest for Recall.
-   kNN is strongest for test Accuracy and Precision.
-   Random Forest has the highest CV F1.
-   Naive Bayes provides competitive overall performance.
-   Logistic Regression provides the strongest overall balance and is
    selected as the final model.

For this dataset, **Logistic Regression is the recommended final
model**, while Decision Tree and Random Forest remain useful
alternatives when the business priority is to maximize churn Recall.
