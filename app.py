import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import requests
import time
import pickle
from fpdf import FPDF
from docx import Document
#from python-docx import Document
from tensorflow.keras.models import load_model
from xgboost import Booster
import xgboost
import tensorflow

# --- Set Page Config (Must be first Streamlit call) ---
st.set_page_config(page_title="Stock Comparison App", layout="wide")

# --- Theme Colors ---
st.markdown(
    """
    <style>
    .main {
        background-color: #e0f7fa;  /* Sky blue */
    }
    .sidebar .sidebar-content {
        background-color: #d8f3dc; /* Light green */
    }
    .css-1q8dd3e {background-color: #fff3cd;}  /* Lemon */
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Title ---
st.title("📈 Stock Comparison and Prediction App")
ticker_symbols = ['NVDA', 'PG', 'PEP', 'RIVN', 'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NFLX']
ticker_prices = [get_stock_price(sym) for sym in ticker_symbols]
ticker_str = " | ".join([f"{sym}: ${price}" for sym, price in zip(ticker_symbols, ticker_prices)])
st.markdown(f'<div class="marquee">🔔 Live Ticker: {ticker_str}</div>', unsafe_allow_html=True)

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Home", "Prediction", "Feature-Based Prediction", "Analysis", "Download Reports"])

# --- Home Tab ---
with tab1:
    st.header("📘 Welcome to the Stock Comparison and Prediction App")
    st.markdown("""
    🌟 This app enables stock price predictions and comparative analysis between primary and competitor stocks.
    Gain insights into market trends, visualize stock comparisons, and receive advisory on selected stocks. 
    
    🌟 Select models, input variables, and enjoy visualizations
        of market trends!

    Features:
    - 🔄 Real-time stock prices
    - 🧠 ML-powered prediction interface
    - 📊 Interactive analysis and advisory
    - 📰 Financial news
    - 📥 Download Excel, PDF, DOC reports
                
    This app lets you analyze and predict stock trends using XGBoost or LSTM models. You can compare stocks, generate predictions, and download comprehensive analysis reports. 
    """)

    st.subheader("📰 Latest News")
    news = get_financial_news()
    if news:
        for article in news[:5]:
            st.markdown(f"**{article['title']}**  \n{article['description']}  \n[Read more]({article['url']})")
            st.markdown("---")
    else:
        st.warning("News unavailable.")

# --- Prediction Tab ---
with tab2:
    st.header("📊 Quick Prediction")

    primary_stock = st.selectbox("Primary Stock", ticker_symbols)
    competitor_stock = st.selectbox("Competitor Stock", ['QUBT', 'JNJ', 'KO', 'TSLA', 'IBM', 'AMD', 'INTC', 'BABA', 'ORCL', 'DIS'])
    model_choice = st.radio("Model", ["XGBoost", "LSTM"])

    st.subheader("🔣 Input Features")
    input_features = {}
    feature_list = ['Open Price','Volume', 'Adj Close', 'Daily Return', 'Tomorrow', 'Cumulative Return',
                    'SMA_20', 'EMA_20', 'RSI', 'MACD', 'Signal_Line', 'Volatility',
                    'Middle_Band', 'Std_Dev']
    for feat in feature_list:
        input_features[feat] = st.number_input(f"{feat}:", 0.0, 100.0, step=0.1, key=feat)

    # Button to trigger prediction
    if st.button("Predict Stock Prices"):
        input_data = np.array([list(input_features.values())])

        if model_choice == "XGBoost":
            predictions_primary = xgb_model_pep.predict(input_data)
            predictions_competitor = xgb_model_ko.predict(input_data)
        elif model_choice == "LSTM":
            predictions_primary = lstm_model_pep.predict(input_data)
            predictions_competitor = lstm_model_ko.predict(input_data)

        st.session_state["results"] = {
            "primary": predictions_primary.flatten(),
            "competitor": predictions_competitor.flatten(),
            "primary_stock": primary_stock,
            "competitor_stock": competitor_stock,
            "advice": "Consider Buying 🚀" if predictions_primary[-1] > predictions_primary[0] else "Hold / Watch ⏸️"
        }

        # Display predictions
        st.write(f"📈 {primary_stock} Predicted Prices:", predictions_primary.flatten())
        st.write(f"📉 {competitor_stock} Predicted Prices:", predictions_competitor.flatten())
        st.info(f"💡 Advisory: {st.session_state['results']['advice']}")

# --- Feature-Based Prediction (Keeps previous logic, can later link to feature-specific models if needed) ---
with tab3:
    st.header("🔍 Predict by Feature Set")
    model_type = st.selectbox("Model Type", ["XGBoost", "LSTM"])

    xgb_feats = ['Adj Close', 'Cummulative Return', 'Tomorrow', 'High', 'Volatility', 'MACD', 'Std_Dev', 'RSI', 'SIgnal_Line']
    lstm_feats = ['Adj Close', 'Tomorrow', 'High', 'Volume', 'RSI', 'SMA_20', 'EMA_20', 'Middle_Band']
    features = xgb_feats if model_type == "XGBoost" else lstm_feats

    feature_inputs = [st.number_input(f"{feat}:", 0.0, 100.0, step=0.1, key=f"{model_type}_{feat}") for feat in features]

    if st.button("🔢 Predict Based on Features"):
        pred = np.round(np.random.uniform(100, 200, 5), 2)
        st.session_state["results"] = {
            "primary": pred,
            "competitor": np.round(pred * np.random.uniform(0.9, 1.1), 2),
            "primary_stock": model_type + "_Model",
            "competitor_stock": "Synthetic",
            "advice": "Trend insight based on feature model"
        }
        st.success(f"Predicted Output ({model_type}): {pred}")

# --- Analysis Tab ---
with tab4:
    st.header("📊 Comparative Analysis")
    result = st.session_state.get("results", {})
    if result:
        df = pd.DataFrame({
            "Day": np.arange(1, len(result['primary']) + 1),
            result['primary_stock']: result['primary'],
            result['competitor_stock']: result['competitor']
        })
        df_melted = df.melt(id_vars="Day", var_name="Stock", value_name="Price")
        fig = px.line(df_melted, x="Day", y="Price", color="Stock", markers=True)
        st.plotly_chart(fig)
        st.success(f"🧠 Advisory: {result['advice']}")
    else:
        st.warning("Run predictions to see analysis.")

# --- Download Reports Tab ---
with tab5:
    st.header("📥 Download Reports")
    result = st.session_state.get("results", {})
    if result:
        df = pd.DataFrame({
            "Day": np.arange(1, len(result['primary']) + 1),
            result['primary_stock']: result['primary'],
            result['competitor_stock']: result['competitor']
        })

        # Excel
        df.to_excel("report.xlsx", index=False)
        with open("report.xlsx", "rb") as f:
            st.download_button("📊 Download Excel", f, file_name="stock_analysis.xlsx")

        # PDF
       # --- PDF (no emojis due to Latin-1 limitation) ---
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Stock Analysis Report", ln=True, align="C")
        pdf.output("analysis.pdf")
    if st.button("Download Document"):
        st.write("Downloading Document...")
        with open("analysis.doc", "w") as f:
            f.write("Analysis Data")
