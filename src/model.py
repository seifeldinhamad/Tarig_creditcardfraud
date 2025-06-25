import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from src.data_processing import clean_data

class FraudDetectionModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()

    def train(self, df, target_column="target"):
        df.columns = df.columns.str.strip().str.lower()
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found.")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )

        # Balance training data
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train_res)
        X_test_scaled = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train_res)

        y_pred = self.model.predict(X_test_scaled)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "auc_score": roc_auc_score(y_test, self.model.predict_proba(X_test_scaled)[:, 1]),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred),
        }

        return metrics, X_test_scaled, y_test

def train_fraud_model(df=None):
    if df is None:
        from src.data_processing import load_fraud_data
        df = load_fraud_data()
        df = clean_data(df)

    fdm = FraudDetectionModel()
    metrics, X_test, y_test = fdm.train(df)
    return fdm, metrics, X_test, y_test
