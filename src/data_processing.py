import pandas as pd
import numpy as np

def load_fraud_data(path="data/balanced_credit_card_data.csv"):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    df.columns = df.columns.str.lower().str.strip()

    # Fill missing numerical values
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].median(), inplace=True)

    # One-hot encode categorical features
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df
