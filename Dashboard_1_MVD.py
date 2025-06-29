import streamlit as st
import pandas as pd
import plotly.express as px

# --- Setup ---
st.set_page_config(page_title="Minimum Viable Dashboard", layout="wide")



# Title and Header
st.title("Credit Card Fraud Dashboard")
st.header("MVD - Minimum Viable Dashboard")

# Load Dataset
DATA_PATH = "data/balanced_credit_card_data.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

data = load_data()

# Sidebar Filters
st.sidebar.header("🔍 Filter Transactions")

# Contract Type Filter
contract_types = st.sidebar.multiselect(
    "Select Contract Type(s)",
    options=data['NAME_CONTRACT_TYPE'].dropna().unique().tolist(),
    default=data['NAME_CONTRACT_TYPE'].dropna().unique().tolist()
)

# Gender Filter
genders = st.sidebar.multiselect(
    "Select Gender(s)",
    options=data['CODE_GENDER'].dropna().unique().tolist(),
    default=data['CODE_GENDER'].dropna().unique().tolist()
)

# AMT_CREDIT Slider
credit_range = st.sidebar.slider(
    "Select AMT_CREDIT Range",
    float(data['AMT_CREDIT'].min()),
    float(data['AMT_CREDIT'].max()),
    (float(data['AMT_CREDIT'].min()), float(data['AMT_CREDIT'].max()))
)

# AMT_INCOME_TOTAL Slider
income_range = st.sidebar.slider(
    "Select AMT_INCOME_TOTAL Range",
    float(data['AMT_INCOME_TOTAL'].min()),
    float(data['AMT_INCOME_TOTAL'].max()),
    (float(data['AMT_INCOME_TOTAL'].min()), float(data['AMT_INCOME_TOTAL'].max()))
)

# Filter Logic
filtered_data = data[
    (data['NAME_CONTRACT_TYPE'].isin(contract_types)) &
    (data['CODE_GENDER'].isin(genders)) &
    (data['AMT_CREDIT'] >= credit_range[0]) & (data['AMT_CREDIT'] <= credit_range[1]) &
    (data['AMT_INCOME_TOTAL'] >= income_range[0]) & (data['AMT_INCOME_TOTAL'] <= income_range[1])
]

# Main Area - Summary Metrics
st.subheader("📊 Summary Metrics")
st.metric("Total Transactions", len(filtered_data))
st.metric("Fraudulent Transactions", int(filtered_data['TARGET'].sum()))
st.metric("Fraud Rate", f"{filtered_data['TARGET'].mean() * 100:.2f}%" if not filtered_data.empty else "0%")

# Preview Filtered Data
st.subheader("📄 Filtered Transactions")
st.dataframe(filtered_data.head(10))


# --------------------
# 📊 Fraud Rate by Gender
# --------------------
st.subheader("📉 Fraud Rate by Gender")
if not filtered_data.empty:
    gender_fraud_rate = filtered_data.groupby("CODE_GENDER")["TARGET"].mean().reset_index()
    gender_fraud_rate["Fraud Rate (%)"] = gender_fraud_rate["TARGET"] * 100

    fig_gender = px.bar(
        gender_fraud_rate,
        x="CODE_GENDER",
        y="Fraud Rate (%)",
        color="CODE_GENDER",
        title="Fraud Rate by Gender",
        labels={"CODE_GENDER": "Gender"},
    )
    st.plotly_chart(fig_gender, use_container_width=True)
else:
    st.info("No data available for gender fraud rate visualization.")

# --------------------
# 📊 Fraud Count by Contract Type
# --------------------
st.subheader("📊 Fraud Count by Contract Type")
if not filtered_data.empty:
    contract_fraud_count = filtered_data[filtered_data["TARGET"] == 1].groupby("NAME_CONTRACT_TYPE")["TARGET"].count().reset_index()
    contract_fraud_count.columns = ["Contract Type", "Fraud Count"]

    fig_contract = px.bar(
        contract_fraud_count,
        x="Contract Type",
        y="Fraud Count",
        color="Contract Type",
        title="Fraud Count by Contract Type",
    )
    st.plotly_chart(fig_contract, use_container_width=True)
else:
    st.info("No data available for contract type fraud count visualization.")


st.caption("⚡️ Filters auto-refresh within ~2 seconds. Built with Streamlit.")
