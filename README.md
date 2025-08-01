# 💳 Credit Card Fraud Detection

This project uses **Machine Learning** to detect fraudulent credit card transactions using **SVM** and **XGBoost** classifiers. It includes **exploratory data analysis (EDA)**, handling of **imbalanced data with SMOTE**, and performance evaluation with metrics such as confusion matrix, precision, recall, and ROC-AUC score.

---

## 📁 Dataset

- Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Transactions made by European cardholders in September 2013.
- Highly imbalanced dataset:
  - **284,807** total transactions
  - **492** fraudulent transactions

---

## ✅ Workflow

1. **Load Data**
2. **EDA (Exploratory Data Analysis)**
3. **Data Preprocessing**
   - Scaling `Amount` and `Time`
   - Splitting dataset
4. **Handling Imbalance**
   - Applied **SMOTE** to oversample minority class
5. **Model Training**
   - **SVM with RBF Kernel**
   - **XGBoost Classifier**
6. **Evaluation Metrics**
   - Confusion Matrix
   - Precision, Recall, F1-Score
   - ROC-AUC Score
7. **Model Saving**
   - Models saved using `joblib`

---

## 📊 Evaluation Metrics

| Metric     | SVM         | XGBoost     |
|------------|-------------|-------------|
| Precision  | ✔️ High      | ✔️ High      |
| Recall     | ✔️ Balanced  | ✔️ Very High |
| ROC-AUC    | ✅ ~0.98+    | ✅ ~0.99+    |

> Note: SVM was trained on a reduced dataset due to performance limitations.

---

## 📦 Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost joblib
~