import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

# --- Setup ---
st.set_page_config(page_title="Executive Fraud Metrics", layout="wide")
st.title("\U0001F4C8 Executive Metrics Dashboard - Fraud Overview")

# --- Load Data ---
@st.cache_data

def load_data():
    df = pd.read_csv("data/balanced_credit_card_data_with_date.csv")
    df["TRANSACTION_DATE"] = pd.to_datetime(df["TRANSACTION_DATE"], errors="coerce")
    df = df.dropna(subset=["TRANSACTION_DATE"])
    return df

df = load_data()

# --- Sidebar Filters ---
st.sidebar.header("\U0001F4C5 Date Range")
min_date, max_date = df["TRANSACTION_DATE"].min(), df["TRANSACTION_DATE"].max()
start_date, end_date = st.sidebar.date_input("Select date range:", [min_date, max_date], min_value=min_date, max_value=max_date)

# Additional Filters
st.sidebar.header("\U0001F50E Filter Transactions")

if "NAME_CONTRACT_TYPE" in df.columns:
    contract_opts = df["NAME_CONTRACT_TYPE"].dropna().unique().tolist()
    selected_contracts = st.sidebar.multiselect("Contract Type", options=contract_opts, default=contract_opts)
    df = df[df["NAME_CONTRACT_TYPE"].isin(selected_contracts)]

if "CODE_GENDER" in df.columns:
    gender_opts = df["CODE_GENDER"].dropna().unique().tolist()
    selected_gender = st.sidebar.multiselect("Gender", options=gender_opts, default=gender_opts)
    df = df[df["CODE_GENDER"].isin(selected_gender)]

if "REGION_RATING_CLIENT" in df.columns:
    region_opts = sorted(df["REGION_RATING_CLIENT"].dropna().unique().tolist())
    selected_regions = st.sidebar.multiselect("Region Rating", options=region_opts, default=region_opts)
    df = df[df["REGION_RATING_CLIENT"].isin(selected_regions)]

# --- Filter by Date ---
df = df[(df["TRANSACTION_DATE"] >= pd.to_datetime(start_date)) & (df["TRANSACTION_DATE"] <= pd.to_datetime(end_date))]

# --- Time KPIs ---
st.subheader("\U0001F552 KPIs by Time Period")
today = datetime.today().date()
yesterday = today - timedelta(days=1)
past_week = today - timedelta(days=7)

# Today
today_df = df[df["TRANSACTION_DATE"].dt.date == today]
week_df = df[df["TRANSACTION_DATE"].dt.date >= past_week]

today_fraud = today_df["TARGET"].sum()
today_rate = (today_fraud / len(today_df) * 100) if len(today_df) else 0

week_fraud = week_df["TARGET"].sum()
week_rate = (week_fraud / len(week_df) * 100) if len(week_df) else 0

colT1, colT2 = st.columns(2)
colT1.metric("Today's Fraud Rate", f"{today_rate:.2f}%", delta=f"{int(today_fraud)} fraud cases")
colT2.metric("Last 7 Days Fraud Rate", f"{week_rate:.2f}%", delta=f"{int(week_fraud)} fraud cases")

# --- Main KPIs ---
st.subheader("\U0001F4CA Key Performance Indicators")
kpi1, kpi2, kpi3 = st.columns(3)
kpi4, kpi5, kpi6 = st.columns(3)

total_tx = len(df)
fraud_tx = df["TARGET"].sum()
fraud_rate = (fraud_tx / total_tx * 100) if total_tx else 0

avg_credit = df["AMT_CREDIT"].mean() if "AMT_CREDIT" in df.columns else 0
fraud_loss = df[df["TARGET"] == 1]["AMT_CREDIT"].sum() if "AMT_CREDIT" in df.columns else 0

gender_note = "N/A"
if "CODE_GENDER" in df.columns:
    fraud_gender = df[df["TARGET"] == 1]["CODE_GENDER"].value_counts(normalize=True) * 100
    if not fraud_gender.empty:
        top_gender = fraud_gender.idxmax()
        gender_rate = fraud_gender.max()
        gender_note = f"{top_gender} ({gender_rate:.1f}%)"

kpi1.metric("Total Transactions", total_tx)
kpi2.metric("Fraudulent Transactions", int(fraud_tx))
kpi3.metric("Fraud Rate", f"{fraud_rate:.2f}%")

kpi4.metric("Avg. Credit Amount", f"${avg_credit:,.0f}")
kpi5.metric("Estimated Fraud Loss", f"${fraud_loss:,.0f}")
kpi6.metric("Top Fraud Gender", gender_note)

# --- Risk Categorization by Income ---
st.subheader("\U0001F4B8 Fraud Risk by Income Band")
if "AMT_INCOME_TOTAL" in df.columns:
    bins = [0, 50000, 100000, 150000, 200000, float('inf')]
    labels = ["<50k", "50k-100k", "100k-150k", "150k-200k", ">200k"]
    df["income_band"] = pd.cut(df["AMT_INCOME_TOTAL"], bins=bins, labels=labels)
    income_summary = df.groupby("income_band").agg(fraud_rate=("TARGET", "mean"), count=("TARGET", "count")).reset_index()
    income_summary["fraud_rate"] = income_summary["fraud_rate"] * 100
    fig_income = px.bar(income_summary, x="income_band", y="fraud_rate", color="count", title="Fraud Rate by Income Band (%)")
    st.plotly_chart(fig_income, use_container_width=True)

# --- Per Capita Fraud by Region ---
st.subheader("\U0001F3E2 Per-Capita Fraud by Region")
if "REGION_RATING_CLIENT" in df.columns:
    region_summary = df.groupby("REGION_RATING_CLIENT").agg(
        total_tx=("TARGET", "count"),
        fraud_tx=("TARGET", "sum")
    ).reset_index()
    region_summary["fraud_per_100_tx"] = (region_summary["fraud_tx"] / region_summary["total_tx"]) * 100
    fig_region = px.bar(region_summary, x="REGION_RATING_CLIENT", y="fraud_per_100_tx",
                        title="Fraud Cases per 100 Transactions by Region")
    st.plotly_chart(fig_region, use_container_width=True)

# --- Time Granularity Trends ---
st.subheader("\U0001F4C6 Temporal Trends")
time_group = st.radio("View Trends by:", ["Daily", "Weekly", "Monthly"], horizontal=True)

if time_group == "Weekly":
    df["period"] = df["TRANSACTION_DATE"].dt.to_period("W").apply(lambda r: r.start_time)
elif time_group == "Monthly":
    df["period"] = df["TRANSACTION_DATE"].dt.to_period("M").apply(lambda r: r.start_time)
else:
    df["period"] = df["TRANSACTION_DATE"].dt.date

summary = df.groupby("period").agg(
    total_tx=("TARGET", "count"),
    fraud_tx=("TARGET", "sum")
).reset_index()
summary["fraud_rate"] = (summary["fraud_tx"] / summary["total_tx"]) * 100

fig1 = px.line(summary, x="period", y="fraud_tx", title=f"{time_group} Fraud Volume", markers=True)
fig2 = px.line(summary, x="period", y="fraud_rate", title=f"{time_group} Fraud Rate (%)", markers=True)

st.plotly_chart(fig1, use_container_width=True)
st.plotly_chart(fig2, use_container_width=True)

# --- Additional Visualization: Fraud by Contract Type ---
if "NAME_CONTRACT_TYPE" in df.columns:
    st.subheader("📄 Fraud Rate by Contract Type")
    contract_summary = df.groupby("NAME_CONTRACT_TYPE").agg(
        fraud_rate=("TARGET", "mean"),
        count=("TARGET", "count")
    ).reset_index()
    contract_summary["fraud_rate"] = contract_summary["fraud_rate"] * 100
    fig_contract = px.bar(contract_summary, x="NAME_CONTRACT_TYPE", y="fraud_rate", color="count",
                          title="Fraud Rate by Contract Type (%)")
    st.plotly_chart(fig_contract, use_container_width=True)

# --- Additional Visualization: Fraud by Gender ---
if "CODE_GENDER" in df.columns:
    st.subheader("🚻 Fraud Rate by Gender")
    gender_summary = df.groupby("CODE_GENDER").agg(
        fraud_rate=("TARGET", "mean"),
        count=("TARGET", "count")
    ).reset_index()
    gender_summary["fraud_rate"] = gender_summary["fraud_rate"] * 100
    fig_gender = px.pie(gender_summary, names="CODE_GENDER", values="count", title="Fraud Distribution by Gender",
                        hover_data=["fraud_rate"], hole=0.4)
    st.plotly_chart(fig_gender, use_container_width=True)

# --- Export ---
st.subheader("\U0001F4E4 Export Summary Report")
csv = summary.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv, f"{time_group.lower()}_fraud_summary.csv", "text/csv")

st.caption("\U0001F6C8 Executive dashboard with advanced KPIs and segmentation filters.")
