import pytest
import sys
import os
from sklearn.metrics import classification_report, confusion_matrix

# Add the `src` directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Import the required functions
from model_pipeline import preprocess_data, train_model

def test_pipeline():
    file_path = r"C:\Users\AZ\Downloads\balanced_credit_card_data.csv"
    # Preprocess data
    data = preprocess_data(file_path)
    assert not data.empty, "Preprocessed data should not be empty"

    # Train model
    model, scaler, X_test_scaled, y_test = train_model(data)

    # Predict and evaluate
    y_pred = model.predict(X_test_scaled)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Assertions for expected metrics
    assert report["0"]["precision"] >= 0.95, "Precision for class 0 is too low"
    assert report["1"]["recall"] >= 0.6, "Recall for class 1 is too low"

    # Print report and confusion matrix for debugging
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
