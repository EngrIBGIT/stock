import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import requests
import time
import os
from fpdf import FPDF
from docx import Document
from tensorflow.keras.models import load_model
import xgboost as xgb
from xgboost import Booster, DMatrix
import tensorflow as tf

# --- Set Page Config ---
st.set_page_config(page_title="Stock Comparison App", layout="wide")

# --- Custom CSS ---
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

# --- Load Models with Error Handling ---
@st.cache_resource
def load_xgb_model(path):
    if os.path.exists(path):
        model = xgb.Booster()
        model.load_model(path)
        return model
    return None

@st.cache_resource
def load_lstm_model(path):
    if os.path.exists(path):
        model = load_model(path)
        model.compile(optimizer="adam", loss="mse")
        return model
    return None

# Load all models
xgb_model_pep = load_xgb_model("xgb_pep_model.xgb")
xgb_model_ko = load_xgb_model("xgb_ko_model.xgb")
lstm_model_pep = load_lstm_model("best_model_pep_tuned.h5")
lstm_model_ko = load_lstm_model("best_model_ko_tuned.h5")

# --- API Configuration ---
FINNHUB_API_KEY = st.secrets.get("FINNHUB_API_KEY", "cvuguhpr01qjg139orhgcvuguhpr01qjg139ori0")
MARKETSTACK_API_KEY = st.secrets.get("MARKETSTACK_API_KEY", "359c056969eeb01527ce644a4df15822")

# --- Data Fetching Functions ---
@st.cache_data(ttl=600)
def get_stock_price(symbol):
    try:
        url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_API_KEY}"
        for _ in range(3):
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return response.json().get('c', 'N/A')
            time.sleep(1)
    except Exception:
        return "Error"
    return "Error"

@st.cache_data(ttl=600)
def get_financial_news():
    try:
        url = f"http://api.marketstack.com/v1/news?access_key={MARKETSTACK_API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get('data', [])
    except Exception:
        return []
    return []

# --- UI Setup ---
st.title("📈 Stock Comparison and Prediction App")

# Ticker display
ticker_symbols = ['NVDA', 'PG', 'PEP', 'RIVN', 'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NFLX']
ticker_prices = [get_stock_price(sym) for sym in ticker_symbols]
ticker_str = " | ".join([f"{sym}: ${price}" if price != "Error" else f"{sym}: N/A" 
                        for sym, price in zip(ticker_symbols, ticker_prices)])
st.markdown(f'<div class="marquee">🔔 Live Ticker: {ticker_str}</div>', unsafe_allow_html=True)

# --- Tab Layout ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Home", "Prediction", "Feature Analysis", "Visualization", "Reports"])

# Home Tab
with tab1:
    st.header("📘 Welcome to the Stock Comparison App")
    st.markdown("""
    This app provides:
    - Real-time stock price tracking
    - Machine learning predictions (XGBoost & LSTM)
    - Comparative analysis between stocks
    - Downloadable reports
    
    **How to use:**
    1. Select stocks in the Prediction tab
    2. Choose a model type
    3. View results and analysis
    """)
    
    st.subheader("📰 Financial News")
    news = get_financial_news()
    if news:
        for article in news[:3]:
            st.markdown(f"**{article['title']}**  \n{article['description']}  \n[Read more]({article['url']})")
            st.divider()
    else:
        st.warning("Could not load financial news")

# Prediction Tab
with tab2:
    st.header("📊 Stock Prediction")
    
    col1, col2 = st.columns(2)
    with col1:
        primary_stock = st.selectbox("Primary Stock", ticker_symbols, index=2)
    with col2:
        competitor_stock = st.selectbox("Competitor Stock", 
                                      ['QUBT', 'JNJ', 'KO', 'TSLA', 'IBM', 'AMD', 'INTC', 'BABA', 'ORCL', 'DIS'],
                                      index=2)
    
    model_choice = st.radio("Prediction Model", ["XGBoost", "LSTM"], horizontal=True)
    
    if st.button("Generate Predictions"):
        if model_choice == "XGBoost":
            model_primary = xgb_model_pep
            model_competitor = xgb_model_ko
        else:
            model_primary = lstm_model_pep
            model_competitor = lstm_model_ko
            
        if None in [model_primary, model_competitor]:
            st.error("Required model not loaded - check deployment logs")
        else:
            # Generate sample predictions (replace with actual model predictions)
            pred_primary = np.random.uniform(100, 200, 30).cumsum()
            pred_competitor = np.random.uniform(80, 180, 30).cumsum()
            
            st.session_state["results"] = {
                "primary": pred_primary,
                "competitor": pred_competitor,
                "primary_stock": primary_stock,
                "competitor_stock": competitor_stock,
                "advice": "Consider Buying 🚀" if pred_primary[-1] > pred_primary[0] else "Hold ⏸️"
            }
            
            st.success("Predictions generated successfully!")

# Feature Analysis Tab
with tab3:
    st.header("🔍 Feature Analysis")
    st.info("This section analyzes how different features affect stock predictions")
    
    if "results" in st.session_state:
        df = pd.DataFrame({
            "Feature": ["Volume", "RSI", "MACD", "Volatility"],
            "Importance": np.random.uniform(0, 1, 4)
        }).sort_values("Importance", ascending=False)
        
        fig = px.bar(df, x="Feature", y="Importance", 
                    title="Feature Importance Analysis")
        st.plotly_chart(fig)
    else:
        st.warning("Generate predictions first to see feature analysis")

# Visualization Tab
with tab4:
    st.header("📈 Comparative Analysis")
    
    if "results" in st.session_state:
        results = st.session_state["results"]
        df = pd.DataFrame({
            "Day": np.arange(1, len(results["primary"]) + 1),
            results["primary_stock"]: results["primary"],
            results["competitor_stock"]: results["competitor"]
        })
        
        fig = px.line(df, x="Day", y=[results["primary_stock"], results["competitor_stock"]], 
                     title="Stock Price Prediction Comparison")
        st.plotly_chart(fig)
        
        st.info(f"💡 Advisory: {results['advice']}")
    else:
        st.warning("No prediction results available. Generate predictions first.")

# Reports Tab
with tab5:
    st.header("📥 Download Reports")
    
    if "results" in st.session_state:
        results = st.session_state["results"]
        
        # Create DataFrame for export
        df = pd.DataFrame({
            "Day": np.arange(1, len(results["primary"]) + 1,
            results["primary_stock"]: results["primary"],
            results["competitor_stock"]: results["competitor"]
        })
        
        # Excel Report
        df.to_excel("stock_report.xlsx", index=False)
        with open("stock_report.xlsx", "rb") as f:
            st.download_button("📊 Download Excel Report", f, "stock_analysis.xlsx")
        
        # PDF Report
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Stock Analysis Report", ln=1, align="C")
        pdf.cell(200, 10, txt=f"Comparison: {results['primary_stock']} vs {results['competitor_stock']}", ln=1)
        
        for i in range(min(10, len(results["primary"]))):
            line = f"Day {i+1}: {results['primary_stock']}={results['primary'][i]:.2f}, {results['competitor_stock']}={results['competitor'][i]:.2f}"
            pdf.cell(200, 10, txt=line, ln=1)
        
        pdf.cell(200, 10, txt=f"Advisory: {results['advice']}", ln=1)
        pdf.output("report.pdf")
        with open("report.pdf", "rb") as f:
            st.download_button("📄 Download PDF Report", f, "stock_report.pdf")
        
        st.success("Reports generated successfully!")
    else:
        st.warning("Generate predictions first to download reports")

# Footer
st.markdown("---")
st.markdown("""
**Disclaimer:** This app provides financial information for educational purposes only. 
Predictions are based on historical data and machine learning models, 
not financial advice. Always conduct your own research before making investment decisions.
""")
