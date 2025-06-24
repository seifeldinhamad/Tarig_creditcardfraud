# src/data_processing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_fraud_data(path="data/balanced_credit_card_data.csv"):
    """
    Load the credit card fraud dataset from the specified CSV file path.
    """
    return pd.read_csv(path)

def clean_data(df):
    """
    Clean and preprocess the dataset:
    - Handle missing values
    - One-hot encode categorical features
    - Create amount category
    """
    # Handle missing numeric values with median imputation
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)

    # Encode categorical variables
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Add an amount category column (for Streamlit filter)
    if 'AMT_CREDIT' in df.columns:
        df['Amount_Category'] = pd.cut(
            df['AMT_CREDIT'],
            bins=[0, 100, 500, 1000, float('inf')],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
    elif 'Amount' in df.columns:
        df['Amount_Category'] = pd.cut(
            df['Amount'],
            bins=[0, 100, 500, 1000, float('inf')],
            labels=['Low', 'Medium', 'High', 'Very High']
        )

    return df

def prepare_features(df, target_column='TARGET'):
    """
    Separate features and labels, apply SMOTE and scale features.
    Returns: X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Apply SMOTE to training data
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_resampled)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_resampled, y_test, scaler
