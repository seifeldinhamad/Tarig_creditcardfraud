import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.data_processing import load_fraud_data, clean_data
from src.model import train_fraud_model
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Performance monitoring
from src.performance_monitor import PerformanceMonitor

# Start monitoring
monitor = PerformanceMonitor()
monitor.start_monitoring()

# Page configuration
st.set_page_config(
    page_title="Fraud Detection Dashboard - Enhanced",
    page_icon="🔒",
    layout="wide"
)

# Sidebar navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.selectbox(
    "Select Page",
    ["Dashboard Overview", "Advanced Analysis", "Model Performance", "Fraud Prediction"]
)

# Load data and model
@st.cache_data
def get_processed_data():
    df = load_fraud_data()
    df = clean_data(df)
    return df

# Load model (this will train if not cached)
with st.spinner("Loading ML model..."):
    model, metrics, X_test, y_test = train_fraud_model()

df = get_processed_data()

# ========== DASHBOARD OVERVIEW PAGE ==========
if page == "Dashboard Overview":
    st.title("🔒 Fraud Detection Dashboard - Enhanced")
    st.markdown("*Sprint 2 - ML-Powered Analytics*")
    st.markdown("---")
    
    # Enhanced filters in sidebar
    st.sidebar.header("🔍 Advanced Filters")
    
    # Filter 1: Transaction Type (enhanced)
    transaction_types = df['Type'].unique().tolist()
    selected_types = st.sidebar.multiselect(
        "Transaction Types",
        options=transaction_types,
        default=transaction_types,
        help="Filter by transaction type"
    )
    
    # Filter 2: Amount range (enhanced)
    min_amount, max_amount = st.sidebar.slider(
        "Amount Range ($)",
        min_value=float(df['Amount'].min()),
        max_value=float(df['Amount'].max()),
        value=(float(df['Amount'].min()), float(df['Amount'].max())),
        help="Filter by transaction amount range"
    )
    
    # Filter 3: Time period
    time_range = st.sidebar.slider(
        "Time Period (hours)",
        min_value=0,
        max_value=48,
        value=(0, 48),
        help="Filter by hour of day"
    )
    
    # Apply enhanced filters
    df['Hour'] = (df['Time'] % 86400) // 3600
    filtered_df = df[
        (df['Type'].isin(selected_types)) & 
        (df['Amount'] >= min_amount) & 
        (df['Amount'] <= max_amount) &
        (df['Hour'] >= time_range[0]) &
        (df['Hour'] <= time_range[1])
    ]
    
    # Enhanced summary metrics
    st.header("📊 Enhanced Summary Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_transactions = len(filtered_df)
        prev_total = len(df)  # comparison baseline
        delta_total = total_transactions - prev_total
        st.metric("Total Transactions", f"{total_transactions:,}", delta=f"{delta_total:,}")
    
    with col2:
        fraud_count = filtered_df['Class'].sum()
        st.metric("Fraud Cases", f"{fraud_count:,}", delta=None)
    
    with col3:
        fraud_rate = (fraud_count / total_transactions * 100) if total_transactions > 0 else 0
        st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
    
    with col4:
        avg_fraud_amount = filtered_df[filtered_df['Class']==1]['Amount'].mean() if fraud_count > 0 else 0
        st.metric("Avg Fraud Amount", f"${avg_fraud_amount:.2f}")
    
    with col5:
        model_accuracy = metrics['accuracy'] * 100
        st.metric("Model Accuracy", f"{model_accuracy:.1f}%")
    
    # Enhanced visualizations
    st.header("📈 Enhanced Analytics")
    
    # Create subplot layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Fraud distribution by hour
        hourly_fraud = filtered_df.groupby(['Hour', 'Class']).size().reset_index(name='Count')
        fig1 = px.bar(hourly_fraud, x='Hour', y='Count', color='Class',
                      title="Fraud Distribution by Hour of Day",
                      color_discrete_map={0: 'lightblue', 1: 'red'})
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Amount vs Time scatter with fraud highlighting
        fig2 = px.scatter(filtered_df.sample(min(1000, len(filtered_df))), 
                         x='Time', y='Amount', color='Class',
                         title="Transaction Amount vs Time",
                         color_discrete_map={0: 'lightblue', 1: 'red'},
                         hover_data=['Type'])
        st.plotly_chart(fig2, use_container_width=True)
    
    # Fraud heatmap by type and amount category
    fraud_heatmap_data = filtered_df.groupby(['Type', 'Amount_Category'])['Class'].agg(['count', 'sum']).reset_index()
    fraud_heatmap_data['fraud_rate'] = (fraud_heatmap_data['sum'] / fraud_heatmap_data['count'] * 100).fillna(0)
    
    pivot_data = fraud_heatmap_data.pivot(index='Type', columns='Amount_Category', values='fraud_rate').fillna(0)
    
    fig3 = px.imshow(pivot_data, 
                     title="Fraud Rate Heatmap (%) by Type and Amount Category",
                     color_continuous_scale='Reds',
                     aspect='auto')
    st.plotly_chart(fig3, use_container_width=True)

# ========== FRAUD PREDICTION PAGE ==========
elif page == "Fraud Prediction":
    st.title("🔮 ML-Powered Fraud Prediction")
    st.markdown("*Enhanced with Random Forest Model*")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input Transaction Details")
        
        with st.form("enhanced_fraud_prediction"):
            # Input fields
            input_amount = st.number_input("Transaction Amount ($)", 
                                         min_value=0.0, value=100.0, step=0.01)
            
            input_type = st.selectbox("Transaction Type", 
                                    options=df['Type'].unique().tolist())
            
            input_time = st.number_input("Time (seconds from start)", 
                                       min_value=0, value=3600, step=1)
            
            # Additional features for better prediction
            v1_input = st.slider("Feature V1", -5.0, 5.0, 0.0, help="Anonymized feature")
            v2_input = st.slider("Feature V2", -5.0, 5.0, 0.0, help="Anonymized feature")
            v3_input = st.slider("Feature V3", -5.0, 5.0, 0.0, help="Anonymized feature")
            
            submitted = st.form_submit_button("🔍 Predict Fraud Risk", type="primary")
    
    with col2:
        if submitted:
            st.subheader("🎯 Prediction Results")
            
            # Prepare input for model
            input_features = np.array([[input_time, input_amount, v1_input, v2_input, v3_input]])
            
            # Scale input using the same scaler used for training
            input_scaled = model.scaler.transform(input_features)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            
            fraud_prob = probability[1] * 100  # Probability of fraud
            
            # Display results with enhanced styling
            if prediction == 1:
                st.error(f"🚨 **HIGH RISK** - Fraud Detected!")
                st.error(f"Fraud Probability: **{fraud_prob:.1f}%**")
            else:
                if fraud_prob > 30:
                    st.warning(f"⚠️ **MEDIUM RISK** - Monitor Transaction")
                    st.warning(f"Fraud Probability: **{fraud_prob:.1f}%**")
                else:
                    st.success(f"✅ **LOW RISK** - Transaction Appears Normal")
                    st.success(f"Fraud Probability: **{fraud_prob:.1f}%**")
            
            # Risk breakdown
            st.subheader("Risk Analysis Breakdown")
            
            # Create gauge chart for probability
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = fraud_prob,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Fraud Probability (%)"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 25], 'color': "lightgreen"},
                        {'range': [25, 50], 'color': "yellow"},
                        {'range': [50, 75], 'color': "orange"},
                        {'range': [75, 100], 'color': "red"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90}}
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Feature importance (if available)
            if hasattr(model.model, 'feature_importances_'):
                st.subheader("Feature Importance")
                feature_names = ['Time', 'Amount', 'V1', 'V2', 'V3']
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.model.feature_importances_
                }).sort_values('Importance', ascending=True)
                
                fig_importance = px.bar(importance_df, x='Importance', y='Feature', 
                                      orientation='h', title="Feature Importance in Fraud Detection")
                st.plotly_chart(fig_importance, use_container_width=True)

# ========== MODEL PERFORMANCE PAGE ==========
elif page == "Model Performance":
    st.title("📊 Model Performance Analytics")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Performance Metrics")
        st.text("Classification Report:")
        st.text(metrics['classification_report'])
        
        st.metric("Model Accuracy", f"{metrics['accuracy']:.3f}")
        st.metric("AUC Score", f"{metrics['auc_score']:.3f}")
    
    with col2:
        st.subheader("Confusion Matrix")
        cm = metrics['confusion_matrix']
        
        fig_cm = px.imshow(cm, 
                          text_auto=True, 
                          aspect="auto",
                          title="Confusion Matrix",
                          labels=dict(x="Predicted", y="Actual"),
                          x=['Normal', 'Fraud'],
                          y=['Normal', 'Fraud'])
        st.plotly_chart(fig_cm, use_container_width=True)

# ========== ADVANCED ANALYSIS PAGE ==========
elif page == "Advanced Analysis":
    st.title("🔬 Advanced Fraud Analysis")
    st.markdown("---")
    
    # Advanced analytics that weren't in MVP
    st.subheader("Fraud Patterns Over Time")
    
    # Time series analysis
    df['Date'] = pd.to_datetime(df['Time'], unit='s')
    daily_fraud = df.groupby([df['Date'].dt.date, 'Class']).size().reset_index(name='Count')
    
    fig_time = px.line(daily_fraud, x='Date', y='Count', color='Class',
                      title="Fraud Cases Over Time",
                      color_discrete_map={0: 'blue', 1: 'red'})
    st.plotly_chart(fig_time, use_container_width=True)
    
    # Statistical analysis
    st.subheader("Statistical Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Amount Statistics by Class:**")
        amount_stats = df.groupby('Class')['Amount'].agg(['mean', 'median', 'std']).round(2)
        st.dataframe(amount_stats)
    
    with col2:
        st.write("**Transaction Type Distribution:**")
        type_dist = df.groupby(['Type', 'Class']).size().unstack(fill_value=0)
        st.dataframe(type_dist)


# Stop monitoring and display
monitor.stop_monitoring()
monitor.display_metrics()

# Footer
st.markdown("---")
st.markdown(f"*Enhanced Dashboard - Sprint 2 | Model Accuracy: {metrics['accuracy']:.1%}*")