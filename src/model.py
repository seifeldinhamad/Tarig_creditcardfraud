import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
import streamlit as st

class FraudDetectionModel:
    def __init__(self, model_type='random_forest'):
        """Initialize fraud detection model"""
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            )
        else:
            self.model = LogisticRegression(
                random_state=42,
                class_weight='balanced'
            )
        
        self.scaler = None
        self.is_trained = False
    
    def train(self, X_train, y_train, use_smote=True):
        """Train the fraud detection model"""
        
        # Apply SMOTE for handling imbalanced data
        if use_smote:
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            st.info(f"SMOTE applied: {len(X_train)} → {len(X_train_balanced)} samples")
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        # Train model
        self.model.fit(X_train_balanced, y_train_balanced)
        self.is_trained = True
        
        return self
    
    def predict(self, X):
        """Make fraud predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get fraud probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': self.model.score(X_test, y_test),
            'auc_score': roc_auc_score(y_test, probabilities),
            'classification_report': classification_report(y_test, predictions),
            'confusion_matrix': confusion_matrix(y_test, predictions)
        }
        
        return metrics
    
    def save_model(self, filepath):
        """Save trained model"""
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath):
        """Load pre-trained model"""
        self.model = joblib.load(filepath)
        self.is_trained = True

@st.cache_resource
def train_fraud_model():
    """Train and cache fraud detection model"""
    from src.data_processing import load_fraud_data, clean_data, prepare_model_data
    
    # Load and prepare data
    df = load_fraud_data()
    df = clean_data(df)
    X_train, X_test, y_train, y_test, scaler = prepare_model_data(df)
    
    # Train model
    model = FraudDetectionModel('random_forest')
    model.train(X_train, y_train, use_smote=True)
    model.scaler = scaler
    
    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    
    return model, metrics, X_test, y_test