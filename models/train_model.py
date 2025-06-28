import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load your existing dataset
df = pd.read_csv("data/balanced_credit_card_data.csv")

# Define features and target
FEATURES = ['AMT_CREDIT', 'AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED']
TARGET = 'TARGET'

# Drop rows with missing values in selected columns
df = df.dropna(subset=FEATURES + [TARGET])

X = df[FEATURES]
y = df[TARGET]

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "fraud_model.pkl")

print("✅ Model trained and saved as fraud_model.pkl")
# Evaluate the model