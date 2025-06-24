#python file CRISP-DM deliverable 1
'''
Project: Fraud Detection Data Mining
Deliverable 1: Code and Documentation

This script follows the CRISP-DM framework, clearly documenting each step:
1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment (placeholder)

Each section includes comments on responsibilities and rationale.
'''

# 1. Business Understanding
# Responsibility: Define the goal of identifying fraudulent credit card transactions.
# Objective: Build and evaluate a classifier to detect fraud.

# 2. Data Understanding
import pandas as pd      # Data manipulation
import numpy as np       # Numerical operations

# 2.1 Load the dataset
# Responsibility: Read CSV file into DataFrame for exploration
data_path = r"C:\Users\AZ\Downloads\balanced_credit_card_data.csv"
data = pd.read_csv(data_path)
print(f"Loaded data with {data.shape[0]} rows and {data.shape[1]} columns")

# 2.2 Inspect dataset
# Responsibility: Examine column names, types, missing values, and basic stats
print("Columns:", data.columns.tolist())
print(data.info())
print(data.head())
print("Missing values per column:\n", data.isnull().sum())

# 3. Data Preparation
# Responsibility: Clean and transform data for modeling

# 3.1 Handle missing numeric values: median imputation
num_cols = data.select_dtypes(include=[np.number]).columns
for col in num_cols:
    median_val = data[col].median()
    data[col].fillna(median_val, inplace=True)
    # Documentation: Using median to mitigate outlier effect

# 3.2 Encode categorical variables: one-hot encoding
cat_cols = data.select_dtypes(include=['object']).columns.tolist()
# Rationale: Convert categories to binaries for algorithm compatibility
data = pd.get_dummies(data, columns=cat_cols, drop_first=True)
print(f"After encoding, dataset has {data.shape[1]} features")

# 3.3 Feature and target separation
# Responsibility: Define X (features) and y (label)
X = data.drop('TARGET', axis=1)
y = data['TARGET']

# 3.4 Train-test split
from sklearn.model_selection import train_test_split
# Responsibility: Split into training and test sets to evaluate generalization
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set: {X_train.shape[0]} samples; Test set: {X_test.shape[0]} samples")

# 3.5 Address class imbalance: SMOTE
from imblearn.over_sampling import SMOTE
# Responsibility: Oversample minority class to balance dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {np.bincount(y_resampled)} (majority, minority)")

# 3.6 Feature scaling: standardization
from sklearn.preprocessing import StandardScaler
# Responsibility: Scale features to zero mean and unit variance
scaler = StandardScaler()
X_resampled_scaled = scaler.fit_transform(X_resampled)
X_test_scaled = scaler.transform(X_test)

# 4. Modeling
# Responsibility: Train machine learning model
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_resampled_scaled, y_resampled)
print("Model training completed.")

# 5. Evaluation
# Responsibility: Assess model performance on test set
from sklearn.metrics import classification_report, confusion_matrix
# 5.1 Make predictions
y_pred = log_reg.predict(X_test_scaled)
# 5.2 Classification metrics
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 6. Deployment
# Placeholder: Code to export model or integrate into a production pipeline
# e.g.,
# import joblib
# joblib.dump(log_reg, 'fraud_model.pkl')

# End of Deliverable 1: All code steps are documented with responsibilities and rationale.
