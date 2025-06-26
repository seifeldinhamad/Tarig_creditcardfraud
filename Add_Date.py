import pandas as pd
import numpy as np
import os

# Get current directory (where script is located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define data paths relative to script location
data_dir = os.path.join(BASE_DIR, "data")
input_path = os.path.join(data_dir, "balanced_credit_card_data.csv")
output_path = os.path.join(data_dir, "balanced_credit_card_data_with_date.csv")

# Ensure 'data' directory exists
os.makedirs(data_dir, exist_ok=True)

# Load the dataset
df = pd.read_csv(input_path)

# Add TRANSACTION_DATE column if it's missing
if "TRANSACTION_DATE" not in df.columns:
    date_range = pd.date_range(end=pd.Timestamp.today(), periods=len(df), freq="h")  # <-- lowercase h
    df["TRANSACTION_DATE"] = date_range

# Save the updated dataset
df.to_csv(output_path, index=False)

print(f"✅ Date column added and saved to: {output_path}")
