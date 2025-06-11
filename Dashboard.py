import streamlit as st
import pandas as pd

# Title and Header
st.title("Credit Card Fraud Dashboard")
st.header("MVP Dashboard")

# Load Dataset
DATA_PATH = "balanced_credit_card_data.csv"
@st.cache
def load_data():
    data = pd.read_csv(DATA_PATH)
    return data

data = load_data()

# Display Dataset
st.subheader("Dataset Preview")
st.dataframe(data.head(10))

# Summary Metrics
st.subheader("Summary Metrics")
total_transactions = data.shape[0]
fraudulent_transactions = data[data['TARGET'] == 1].shape[0]
non_fraudulent_transactions = total_transactions - fraudulent_transactions

st.metric("Total Transactions", total_transactions)
st.metric("Fraudulent Transactions", fraudulent_transactions)
st.metric("Non-Fraudulent Transactions", non_fraudulent_transactions)

# Filter by Amount
st.subheader("Filter Transactions by Amount")
amount_range = st.slider("Select Range of Amount", 
                         min_value=float(data['AMT_CREDIT'].min()), 
                         max_value=float(data['AMT_CREDIT'].max()),
                         value=(float(data['AMT_CREDIT'].min()), float(data['AMT_CREDIT'].max())))

filtered_data = data[(data['AMT_CREDIT'] >= amount_range[0]) & (data['AMT_CREDIT'] <= amount_range[1])]
st.write(f"Filtered Transactions: {filtered_data.shape[0]}")
st.dataframe(filtered_data)

# Closing Note
st.text("This is a basic MVP dashboard for credit card fraud detection.")
