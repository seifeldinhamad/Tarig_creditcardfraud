import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def preprocess_data(file_path):
    data = pd.read_csv(file_path)

    # Impute missing numeric values
    data[data.select_dtypes(include=[np.number]).columns] = \
        data.select_dtypes(include=[np.number]).fillna(data.select_dtypes(include=[np.number]).median())

    # One-hot encode categoricals
    data = pd.get_dummies(data, drop_first=True)
    return data

def train_model(data, target_column='TARGET'):
    # Split features and target
    y = data[target_column]
    X = data.drop(target_column, axis=1)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Balance training data with SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Scale features
    scaler = StandardScaler()
    X_resampled_scaled = scaler.fit_transform(X_resampled)
    X_test_scaled = scaler.transform(X_test)

    # Train logistic regression
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_resampled_scaled, y_resampled)

    return model, scaler, X_test_scaled, y_test
