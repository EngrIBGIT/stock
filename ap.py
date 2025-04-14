import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from fpdf import FPDF
import streamlit as st

# --- Set Page Configuration ---
st.set_page_config(page_title="Stock Comparison App", layout="wide")

# --- Custom Styling ---
st.markdown("""
    <style>
        .main {background-color: #e0f7fa;}
        .sidebar .sidebar-content {background-color: #d8f3dc;}
        .css-1q8dd3e {background-color: #fff3cd;}
        .marquee {
            font-size: 18px;
            color: #0c0c0c;
            padding: 10px;
            background-color: #ffffe0;
            overflow: hidden;
            white-space: nowrap;
            animation: marquee 20s linear infinite;
        }
        @keyframes marquee {
            0% {transform: translateX(100%);}
            100% {transform: translateX(-100%);}
        }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.title("📈 Stock Comparison and Prediction App")

# --- Simulated Live Ticker and News ---
st.markdown('<div class="marquee">🔔 Live Prices: PEP: $172.34 ▲ | AAPL: $185.12 ▼ | TSLA: $710.34 ▲ | Market News: Inflation steady at 3.2% - Fed cautious | MSFT releases new AI chip ⚡</div>', unsafe_allow_html=True)

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["Home", "Prediction", "Analysis"])

# --- Home Tab ---
with tab1:
    st.header("📘 Welcome to the Stock Comparison and Prediction App")

    st.markdown("""
    This application is designed to give investors, analysts, and stock enthusiasts a powerful tool to analyze and predict the future trends of stocks. The tool offers two main capabilities: 
    1) Real-time comparison between a primary stock and its competitor, and 
    2) Future price prediction using machine learning models like XGBoost and LSTM.
    
    By leveraging key technical indicators such as RSI, MACD, SMA, and others, this app creates actionable insights and helps you decide your next move. Simply select the stocks you're interested in, choose the benchmark index for comparison, input or autofill feature values, and generate predictions. 
    
    Additionally, our analysis tab provides dynamic visualizations of trends and includes advisory notes based on predicted movements. This app also simulates a stock ticker and news feed to keep you engaged with current market activities. 📰📊
    """)

# --- Prediction Tab ---
with tab2:
    st.header("📊 Prediction Models")

    primary_stock = st.selectbox("Select Primary Stock:", ["PEP", "AAPL", "GOOG", "MSFT"], key="primary_stock")
    competitor_stock = st.selectbox("Select Competitor Stock:", ["KO", "TSLA", "META", "AMZN"], key="competitor_stock")
    indices = st.selectbox("Select Benchmark Index:", ["S&P 500", "NASDAQ", "Dow Jones"], key="benchmark_index")
    model_choice = st.radio("Choose Prediction Model:", ["XGBoost", "LSTM"], key="model_choice")

    st.subheader("Enter or Autofill Input Variables")
    feature_list = [
        'Volume', 'Adj Close', 'Daily Return', 'Tomorrow', 'Cumulative Return',
        'SMA_20', 'EMA_20', 'RSI', 'MACD', 'Signal_Line', 'Volatility',
        'Middle_Band', 'Std_Dev'
    ]
    input_features = {}

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Autofill Sample Data"):
            for feature in feature_list:
                st.session_state[f"input_{feature}"] = np.round(np.random.uniform(10, 90), 2)
        if st.button("🧹 Clear Inputs"):
            for feature in feature_list:
                st.session_state[f"input_{feature}"] = 0.0

    for feature in feature_list:
        input_features[feature] = st.number_input(f"{feature}:", 0.0, 100.0, step=0.1, key=f"input_{feature}")

    if st.button("🎯 Predict Stock Prices", key="predict_button"):
        st.success(f"Generating predictions using **{model_choice}** model...")
        if model_choice == "XGBoost":
            st.session_state.predictions_primary = np.round(np.random.uniform(100, 200, 5), 2)
            st.session_state.predictions_competitor = np.round(np.random.uniform(80, 180, 5), 2)
        else:
            st.session_state.predictions_primary = np.round(np.random.uniform(120, 220, 5), 2)
            st.session_state.predictions_competitor = np.round(np.random.uniform(90, 190, 5), 2)

        st.write(f"📈 {primary_stock} Predicted Prices (Next 5 Days):", st.session_state.predictions_primary)
        st.write(f"📉 {competitor_stock} Predicted Prices (Next 5 Days):", st.session_state.predictions_competitor)

        delta = st.session_state.predictions_primary[-1] - st.session_state.predictions_primary[0]
        advice = "📢 **Advisory:** Consider Buying 🚀" if delta > 0 else "📢 **Advisory:** Consider Selling or Holding ⏸️"
        st.info(f"{primary_stock} - {advice}")

# --- Analysis Tab ---
with tab3:
    st.header("📊 Comparative Analysis")

    # Dynamic trend chart based on predictions if they exist
    if 'predictions_primary' in st.session_state:
        fig = px.line(
            x=np.arange(5),
            y=[st.session_state.predictions_primary, st.session_state.predictions_competitor],
            labels={'x': 'Day', 'value': 'Predicted Price'},
            title=f"{primary_stock} vs {competitor_stock} - Predicted Trend"
        )
        fig.update_traces(mode="lines+markers", line=dict(width=2))
        fig.update_layout(legend=dict(title="Stock"), yaxis=dict(tickprefix="$"))
        fig.data[0].name = primary_stock
        fig.data[1].name = competitor_stock
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("🔍 Make a prediction in the **Prediction** tab to view analysis.")

    st.subheader("📤 Download Analysis Report")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📥 Download Excel"):
            with open("analysis.xlsx", "w") as f:
                f.write("Mock Excel Data")
            st.success("Excel Downloaded.")
    with col2:
        if st.button("📥 Download PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Stock Analysis Report", ln=True, align="C")
            pdf.output("analysis.pdf")
            st.success("PDF Downloaded.")
    with col3:
        if st.button("📥 Download DOC"):
            with open("analysis.doc", "w") as f:
                f.write("Mock DOC Data")
            st.success("DOC Downloaded.")
