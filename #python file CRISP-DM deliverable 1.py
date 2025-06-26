#python file CRISP-DM deliverable 1
'''
Project: Fraud Detection Data Mining
Deliverable 1: Code and Documentation

This script follows the CRISP-DM framework, clearly documenting each step:
1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation and Comparison
6. Deployment
'''

# 1. Business Understanding
# Responsibility: Define the goal of identifying fraudulent credit card transactions.
# Objective: Build and evaluate classifiers to detect fraud.

# 2. Data Understanding
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

# 2.1 Load the dataset
data_path = "./data/balanced_credit_card_data.csv"
data = pd.read_csv(data_path)
print(f"Loaded data with {data.shape[0]} rows and {data.shape[1]} columns")

# 2.2 Inspect dataset
print("Columns:", data.columns.tolist())
print(data.info())
print(data.head())
print("Missing values per column:\n", data.isnull().sum())

# 3. Data Preparation
# 3.1 Handle missing numeric values: median imputation
num_cols = data.select_dtypes(include=[np.number]).columns
for col in num_cols:
    median_val = data[col].median()
    data[col].fillna(median_val, inplace=True)

# 3.2 Encode categorical variables: one-hot encoding
cat_cols = data.select_dtypes(include=['object']).columns.tolist()
data = pd.get_dummies(data, columns=cat_cols, drop_first=True)
print(f"After encoding, dataset has {data.shape[1]} features")

# 3.3 Feature and target separation
X = data.drop('TARGET', axis=1)
y = data['TARGET']

# 3.4 Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set: {X_train.shape[0]} samples; Test set: {X_test.shape[0]} samples")

# 3.5 Address class imbalance: SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {np.bincount(y_resampled)} (majority, minority)")

# 3.6 Feature scaling
scaler = StandardScaler()
X_resampled_scaled = scaler.fit_transform(X_resampled)
X_test_scaled = scaler.transform(X_test)

# 4. Modeling
# Logistic Regression
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_resampled_scaled, y_resampled)
print("Logistic Regression training completed.")

# Random Forest (fit on original scale as tree-based models don't require scaling)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
print("Random Forest training completed.")

# 5. Evaluation and Comparison
# Logistic Regression Evaluation
y_pred_lr = log_reg.predict(X_test_scaled)
acc_lr = accuracy_score(y_test, y_pred_lr)
prec_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)

print("\n--- Logistic Regression Evaluation ---")
print("Classification Report:\n", classification_report(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

# Random Forest Evaluation
y_pred_rf = rf_model.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

print("\n--- Random Forest Evaluation ---")
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# Compare Results
print("\n--- Model Comparison Summary ---")
print(f"Accuracy     | Logistic Regression: {acc_lr:.4f} | Random Forest: {acc_rf:.4f}")
print(f"Precision    | Logistic Regression: {prec_lr:.4f} | Random Forest: {prec_rf:.4f}")
print(f"Recall       | Logistic Regression: {recall_lr:.4f} | Random Forest: {recall_rf:.4f}")
print(f"F1 Score     | Logistic Regression: {f1_lr:.4f} | Random Forest: {f1_rf:.4f}")

# Select Best Model
if f1_rf > f1_lr:
    best_model = rf_model
    model_name = "Random Forest"
else:
    best_model = log_reg
    model_name = "Logistic Regression"

print(f"\n✅ Best Performing Model: {model_name}")

# 6. Deployment: Save best model
joblib.dump(best_model, 'fraud_model.pkl')
print("Model exported as fraud_model.pkl")

# End of Deliverable 1
