import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# --- Configuration ---
st.set_page_config(page_title="Fraud Risk Scoring", layout="wide")
st.title("🚨 Real-Time Fraud Risk Scoring Dashboard")

# --- Load Data ---
@st.cache_data
def load_data():
    return pd.read_csv("balanced_credit_card_data.csv")

data = load_data()

# --- Load Model ---
@st.cache_resource
def load_model():
    return joblib.load("fraud_model.pkl")

model = load_model()

# --- Select Features for Prediction ---
FEATURES = ['AMT_CREDIT', 'AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED']  # Adjust as needed

# --- Risk Thresholds ---
def categorize_risk(prob):
    if prob >= 0.75:
        return "High"
    elif prob >= 0.4:
        return "Medium"
    else:
        return "Low"

# --- Prediction + Timer ---
st.subheader("🔎 Risk Prediction Table")

start_time = time.time()
X = data[FEATURES]
probs = model.predict_proba(X)[:, 1]
latency = time.time() - start_time

data["Fraud_Risk_Score"] = np.round(probs, 3)
data["Risk_Category"] = data["Fraud_Risk_Score"].apply(categorize_risk)
data["Flag_for_Followup"] = False

# --- Filtering ---
risk_filter = st.multiselect(
    "Filter by Risk Category",
    options=["High", "Medium", "Low"],
    default=["High", "Medium", "Low"]
)

filtered = data[data["Risk_Category"].isin(risk_filter)]

# --- Display Table with Flagging ---
edited = st.data_editor(
    filtered[["SK_ID_CURR", "AMT_CREDIT", "AMT_INCOME_TOTAL", "Fraud_Risk_Score", "Risk_Category", "Flag_for_Followup"]],
    use_container_width=True,
    key="risk_table",
)

# --- Performance Metric ---
st.caption(f"⏱️ Inference latency: {latency:.3f} seconds")

# --- Export Flagged Transactions ---
flagged = edited[edited["Flag_for_Followup"] == True]
if not flagged.empty:
    st.subheader("✅ Flagged Transactions for Follow-up")
    st.dataframe(flagged)
    csv = flagged.to_csv(index=False).encode("utf-8")
    st.download_button("Download Flagged Transactions CSV", csv, "flagged_transactions.csv", "text/csv")

st.caption("Real-time model inference and fraud prioritization. Powered by Streamlit + scikit-learn.")
