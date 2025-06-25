import streamlit as st
import pandas as pd
from src.data_processing import load_fraud_data, clean_data
from src.model import train_fraud_model
import plotly.express as px

# Configure Streamlit
st.set_page_config(page_title="Credit Card Fraud Dashboard", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = load_fraud_data()
    df = clean_data(df)
    return df

df = load_data()

# Train model
with st.spinner("Training fraud detection model..."):
    model, metrics, X_test, y_test = train_fraud_model(df)

# Sidebar
st.sidebar.title("Filters")
selected_contracts = st.sidebar.multiselect(
    "Select Contract Types",
    options=df.columns[df.columns.str.startswith("name_contract_type_")].tolist(),
    default=[],
)

# Filter logic
filtered_df = df.copy()
if selected_contracts:
    filtered_df = filtered_df[df[selected_contracts].any(axis=1)]

# Main page
st.title("🔍 Credit Card Fraud Detection Dashboard")
st.metric("Total Records", len(filtered_df))

if "target" in filtered_df.columns:
    st.metric("Fraud Cases", int(filtered_df["target"].sum()))
    st.metric("Fraud Rate", f"{filtered_df['target'].mean() * 100:.2f}%")

# Plot
if "amt_credit" in filtered_df.columns and "target" in filtered_df.columns:
    fig = px.histogram(filtered_df, x="amt_credit", color="target", barmode="overlay",
                       title="Credit Amount Distribution by Fraud")
    st.plotly_chart(fig, use_container_width=True)
