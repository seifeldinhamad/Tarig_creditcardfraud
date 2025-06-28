import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components
from pyvis.network import Network
import networkx as nx

# --- Configuration ---
st.set_page_config(page_title="Fraud Risk Scoring", layout="wide")
st.title("🚨 Real-Time Fraud Risk Scoring Dashboard")

# --- Load Data ---
@st.cache_data
def load_data():
    return pd.read_csv("data/balanced_credit_card_data.csv")

data = load_data()

# --- Load Model ---
@st.cache_resource
def load_model():
    return joblib.load("models/fraud_model.pkl")

model = load_model()

# --- Select Features for Prediction ---
FEATURES = ['AMT_CREDIT', 'AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED']

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

# --- Visualization 1: Scatter Plot ---
st.subheader("📊 Risk Distribution: Credit vs Income")
fig_scatter = px.scatter(
    filtered,
    x="AMT_INCOME_TOTAL",
    y="AMT_CREDIT",
    color="Risk_Category",
    size="Fraud_Risk_Score",
    hover_data=["SK_ID_CURR"],
    title="Credit Amount vs Income Colored by Risk Category"
)
st.plotly_chart(fig_scatter, use_container_width=True)

# --- Visualization 2: Bar Chart ---
st.subheader("📊 Fraud Risk Category Count")
risk_counts = data["Risk_Category"].value_counts().reset_index()
risk_counts.columns = ["Risk Category", "Count"]
fig_bar = px.bar(
    risk_counts,
    x="Risk Category",
    y="Count",
    color="Risk Category",
    title="Distribution of Risk Categories"
)
st.plotly_chart(fig_bar, use_container_width=True)

# --- Visualization 3: Heatmap ---
st.subheader("🧫 Feature Correlation Heatmap")
corr_matrix = filtered[FEATURES + ['Fraud_Risk_Score']].corr()
fig_corr, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
st.pyplot(fig_corr)

# --- Visualization 4: Network Graph ---
st.subheader("🌐 Network Graph of High-Risk Transactions")
high_risk = filtered[filtered["Risk_Category"] == "High"].copy()
G = nx.Graph()
for idx, row in high_risk.iterrows():
    income_group = int(row["AMT_INCOME_TOTAL"] // 10000)
    income_range = f"{income_group * 10000}-{(income_group + 1) * 10000 - 1}"
    income_node = f"Income {income_range}"
    G.add_node(row["SK_ID_CURR"], title=f"Risk Score: {row['Fraud_Risk_Score']}")
    G.add_node(income_node, title=f"Income Band: {income_range}")
    G.add_edge(income_node, row["SK_ID_CURR"])

net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white")
net.from_nx(G)
net.show_buttons(filter_=['physics'])
net.save_graph("network_graph.html")
components.html(open("network_graph.html", 'r').read(), height=550)

st.caption("Real-time model inference and fraud prioritization. Powered by Streamlit + scikit-learn.")
