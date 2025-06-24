'''
Test Script: Fraud Detection Model Testing
This script tests each step of the project pipeline, including:
1. Data Loading
2. Preprocessing
3. Model Training
4. Model Evaluation
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# 1. Load the Dataset
def load_data(filepath):
    print("Loading dataset...")
    data = pd.read_csv(filepath)
    print(f"Loaded data with {data.shape[0]} rows and {data.shape[1]} columns")
    return data

# 2. Preprocess Data
def preprocess_data(data):
    print("Preprocessing dataset...")
    # Handle missing numeric values
    num_cols = data.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        data[col].fillna(data[col].median(), inplace=True)
    
    # Encode categorical variables
    cat_cols = data.select_dtypes(include=['object']).columns
    data = pd.get_dummies(data, columns=cat_cols, drop_first=True)
    
    # Separate features and target
    X = data.drop('TARGET', axis=1)
    y = data['TARGET']
    print(f"Dataset after preprocessing: {X.shape[1]} features")
    return X, y

# 3. Train-Test Split and SMOTE
def split_and_balance_data(X, y):
    print("Splitting and balancing data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {np.bincount(y_resampled)} (majority, minority)")
    return X_resampled, X_test, y_resampled, y_test

# 4. Scale Data
def scale_data(X_train, X_test):
    print("Scaling data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# 5. Train Model
def train_model(X_train_scaled, y_train):
    print("Training model...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    print("Model training completed.")
    return model

# 6. Evaluate Model
def evaluate_model(model, X_test_scaled, y_test):
    print("Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Main Test Code
if __name__ == "__main__":
    # Define file path
    data_path = r"C:\Users\AZ\Downloads\balanced_credit_card_data.csv"
    
    # Execute steps
    data = load_data(data_path)
    X, y = preprocess_data(data)
    X_resampled, X_test, y_resampled, y_test = split_and_balance_data(X, y)
    X_train_scaled, X_test_scaled = scale_data(X_resampled, X_test)
    model = train_model(X_train_scaled, y_resampled)
    evaluate_model(model, X_test_scaled, y_test)
