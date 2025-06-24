import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.data_processing import load_fraud_data, clean_data
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Fraud Detection Dashboard - MVP",
    page_icon="🔒",
    layout="wide"
)

# Title
st.title("🔒 Fraud Detection Dashboard - MVP")
st.markdown("---")

# Load and process data
@st.cache_data
def get_processed_data():
    df = load_fraud_data()
    df = clean_data(df)
    return df

df = get_processed_data()

# Fix missing columns if necessary
if 'Type' not in df.columns:
    df['Type'] = 'Online'  # Default dummy type for all transactions

if 'Class' not in df.columns and 'TARGET' in df.columns:
    df = df.rename(columns={'TARGET': 'Class'})

if 'Amount' not in df.columns and 'AMT_CREDIT' in df.columns:
    df = df.rename(columns={'AMT_CREDIT': 'Amount'})

# Sidebar filters (Use Case 1: Filter by type and amount)
st.sidebar.header("🔍 Filters")

# Filter 1: Transaction Type
transaction_types = df['Type'].unique().tolist()
selected_types = st.sidebar.multiselect(
    "Select Transaction Types",
    options=transaction_types,
    default=transaction_types
)

# Filter 2: Amount Category
amount_categories = df['Amount_Category'].unique().tolist()
selected_amounts = st.sidebar.multiselect(
    "Select Amount Categories",
    options=amount_categories,
    default=amount_categories
)

# Apply filters
filtered_df = df[
    (df['Type'].isin(selected_types)) & 
    (df['Amount_Category'].isin(selected_amounts))
]

# Use Case 3: Summary Metrics
st.header("📊 Summary Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_transactions = len(filtered_df)
    st.metric("Total Transactions", f"{total_transactions:,}")

with col2:
    fraud_count = filtered_df['Class'].sum()
    st.metric("Fraud Cases", f"{fraud_count:,}")

with col3:
    fraud_rate = (fraud_count / total_transactions * 100) if total_transactions > 0 else 0
    st.metric("Fraud Rate", f"{fraud_rate:.2f}%")

with col4:
    total_amount = filtered_df['Amount'].sum()
    st.metric("Total Amount", f"${total_amount:,.2f}")

# Use Case 2: Simple Fraud Prediction (Rule-based for MVP)
st.header("🔮 Fraud Prediction - MVP")

st.markdown("**Simple Rule-Based Prediction (MVP Version)**")

# Input form for prediction
with st.form("fraud_prediction"):
    col1, col2 = st.columns(2)
    
    with col1:
        input_amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)
        input_type = st.selectbox("Transaction Type", options=transaction_types)
    
    with col2:
        input_time = st.number_input("Time (seconds from start)", min_value=0, value=3600)
    
    submitted = st.form_submit_button("Predict Fraud Risk")
    
    if submitted:
        # Simple rule-based prediction for MVP
        risk_score = 0
        
        # Rule 1: High amounts are riskier
        if input_amount > 1000:
            risk_score += 30
        elif input_amount > 500:
            risk_score += 15
        
        # Rule 2: Certain transaction types are riskier
        if input_type == 'Online':
            risk_score += 20
        elif input_type == 'ATM':
            risk_score += 10
        
        # Rule 3: Unusual times (very late/early) are riskier
        hour = (input_time % 86400) // 3600
        if hour < 6 or hour > 22:
            risk_score += 25
        
        # Determine risk level
        if risk_score > 50:
            risk_level = "HIGH"
            risk_color = "red"
        elif risk_score > 25:
            risk_level = "MEDIUM"
            risk_color = "orange"
        else:
            risk_level = "LOW"
            risk_color = "green"
        
        st.markdown(f"**Risk Score:** {risk_score}/100")
        st.markdown(f"**Risk Level:** :{risk_color}[{risk_level}]")

# Use Case 3: Basic Visualization
st.header("📈 Transaction Analysis")

# Fraud distribution by type
fraud_by_type = filtered_df.groupby(['Type', 'Class']).size().reset_index(name='Count')
fig1 = px.bar(fraud_by_type, x='Type', y='Count', color='Class',
              title="Transactions by Type and Fraud Status",
              color_discrete_map={0: 'lightblue', 1: 'red'})
st.plotly_chart(fig1, use_container_width=True)

# Amount distribution
fig2 = px.histogram(filtered_df, x='Amount', color='Class',
                    title="Transaction Amount Distribution",
                    nbins=50,
                    color_discrete_map={0: 'lightblue', 1: 'red'})
st.plotly_chart(fig2, use_container_width=True)

# Data preview
st.header("📋 Filtered Data Preview")
st.dataframe(filtered_df.head(100))

# Footer
st.markdown("---")
st.markdown("*MVP Dashboard - Sprint 1*")
