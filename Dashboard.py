import streamlit as st
import pandas as pd

# Title and Header
st.title("Credit Card Fraud Dashboard")
st.header("MVP Dashboard")

# Load Dataset
DATA_PATH = "balanced_credit_card_data.csv"
@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    return data

data = load_data()

# Dataset Preview with Adjustable Rows
st.subheader("Dataset Preview")
# Add a slider to select the number of rows to display
rows_to_display = st.slider("Select number of rows to display", 1, len(data), 10)
st.dataframe(data.head(rows_to_display))  # Update this section to replace fixed preview rows

# Summary Metrics
st.subheader("Summary Metrics")
total_transactions = data.shape[0]
fraudulent_transactions = data[data['TARGET'] == 1].shape[0]
non_fraudulent_transactions = total_transactions - fraudulent_transactions

st.metric("Total Transactions", total_transactions)
st.metric("Fraudulent Transactions", fraudulent_transactions)
st.metric("Non-Fraudulent Transactions", non_fraudulent_transactions)

# Filter Transactions
st.subheader("Filter Transactions")
# Add radio buttons for filtering
filter_option = st.radio("Filter by:", ("All", "Fraudulent", "Non-Fraudulent"))

# Add a Filter by Amount
st.subheader("Filter Transactions by Amount")
amount_range = st.slider("Select Range of Amount", 
                         min_value=float(data['AMT_CREDIT'].min()), 
                         max_value=float(data['AMT_CREDIT'].max()),
                         value=(float(data['AMT_CREDIT'].min()), float(data['AMT_CREDIT'].max())))

filtered_data = data[(data['AMT_CREDIT'] >= amount_range[0]) & (data['AMT_CREDIT'] <= amount_range[1])]

# Apply the selected filter
if filter_option == "Fraudulent":
    filtered_data = filtered_data[filtered_data['TARGET'] == 1]
elif filter_option == "Non-Fraudulent":
    filtered_data = filtered_data[filtered_data['TARGET'] == 0]

st.write(f"Filtered Transactions: {filtered_data.shape[0]}")
st.dataframe(filtered_data)

# Closing Note
st.text("This is a basic MVP dashboard for credit card fraud detection.")
