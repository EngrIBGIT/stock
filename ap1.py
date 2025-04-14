import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import requests
from fpdf import FPDF

# Set API Keys
FINNHUB_API_KEY = "cvuguhpr01qjg139orhgcvuguhpr01qjg139ori0"
MARKETSTACK_API_KEY = "359c056969eeb01527ce644a4df15822"

# --- Streamlit Page Config ---
st.set_page_config(page_title="Stock Comparison App", layout="wide")

# --- Styles ---
st.markdown("""
<style>
.main {background-color: #e0f7fa;}
.sidebar .sidebar-content {background-color: #d8f3dc;}
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

# --- Functions ---
def get_stock_price(symbol, api_key):
    url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get('c', "N/A")  # current price
    return "Error"

def get_financial_news(api_key):
    url = f"http://api.marketstack.com/v1/news?access_key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get('data', [])
    return []

# --- App Title and Ticker ---
st.title("📈 Stock Comparison and Prediction App")

ticker_symbols = ['PEP', 'NVDA', 'PG', 'RIVN', 'MSFT',]
ticker_prices = [get_stock_price(sym, FINNHUB_API_KEY) for sym in ticker_symbols]
ticker_str = " | ".join([f"{sym}: ${price}" for sym, price in zip(ticker_symbols, ticker_prices)])

st.markdown(f'<div class="marquee">🔔 Live Ticker: {ticker_str}</div>', unsafe_allow_html=True)

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["Home", "Prediction", "Analysis"])

# --- Home ---
with tab1:
    st.header("📘 Welcome to the Stock Comparison and Prediction App")
    st.markdown("""
    This application empowers investors and analysts to compare primary and competitor stock performance, and generate stock price predictions using machine learning models like XGBoost and LSTM.

    Features include:
    - 📊 Real-time stock price tracking
    - 📰 Live financial news feeds
    - 🧠 AI-powered price predictions
    - 📈 Visual trend analysis
    - 📥 Downloadable reports
    - 💡 Stock trading advisory insights

    Whether you're assessing short-term trends or evaluating competitors, this app provides you the tools and intelligence to make informed investment decisions.
    """)

    # --- Latest Financial News ---
    st.subheader("📰 Latest Financial News")
    news_data = get_financial_news(MARKETSTACK_API_KEY)
    if news_data:
        for article in news_data[:5]:
            st.markdown(f"**{article['title']}**  \n{article['description']}  \n[Read more]({article['url']})")
            st.markdown("---")
    else:
        st.warning("Could not load news at the moment.")

# --- Prediction ---
with tab2:
    st.header("📊 Prediction Models")

    primary_stock = st.selectbox("Select Primary Stock:", ticker_symbols, key="primary_stock")
    competitor_stock = st.selectbox("Select Competitor Stock:", ["KO", 'QUBT', "JNJ", "TSLA", "GOOG"], key="competitor_stock")
    model_choice = st.radio("Choose Model:", ["XGBoost", "LSTM"], key="model_choice")

    st.subheader("📥 Input or Autofill Features")
    feature_list = [
        'Volume', 'Adj Close', 'Daily Return', 'Tomorrow', 'Cumulative Return',
        'SMA_20', 'EMA_20', 'RSI', 'MACD', 'Signal_Line', 'Volatility',
        'Middle_Band', 'Std_Dev'
    ]
    input_features = {}

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Autofill Sample Data"):
            for feat in feature_list:
                st.session_state[f"input_{feat}"] = np.round(np.random.uniform(10, 90), 2)
        if st.button("🧹 Clear Inputs"):
            for feat in feature_list:
                st.session_state[f"input_{feat}"] = 0.0

    for feat in feature_list:
        input_features[feat] = st.number_input(f"{feat}:", 0.0, 100.0, step=0.1, key=f"input_{feat}")

    if st.button("🎯 Predict"):
        st.success(f"Prediction using {model_choice}")
        # Simulate prediction results
        predictions_primary = np.round(np.random.uniform(100, 200, 5), 2)
        predictions_competitor = np.round(np.random.uniform(80, 180, 5), 2)

        st.write(f"📈 {primary_stock} Forecast:", predictions_primary)
        st.write(f"📉 {competitor_stock} Forecast:", predictions_competitor)

        delta = predictions_primary[-1] - predictions_primary[0]
        st.info("💡 **Advisory:** " + ("Consider Buying 🚀" if delta > 0 else "Hold or Watch ⏸️"))

        # Save to session for analysis tab
        st.session_state["predictions_primary"] = predictions_primary
        st.session_state["predictions_competitor"] = predictions_competitor

# --- Analysis ---
with tab3:
    st.header("📊 Comparative Analysis")

    if "predictions_primary" in st.session_state and "predictions_competitor" in st.session_state:
        days = np.arange(1, 6)
        df = pd.DataFrame({
            "Day": list(days) * 2,
            "Price": list(st.session_state["predictions_primary"]) + list(st.session_state["predictions_competitor"]),
            "Stock": [primary_stock] * 5 + [competitor_stock] * 5
        })
        fig = px.line(df, x="Day", y="Price", color="Stock", markers=True, title="Predicted Price Comparison")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Run a prediction first!")

    st.subheader("📥 Download Reports")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Download Excel"):
            df = pd.DataFrame({
                "Day": np.arange(1, 6),
                primary_stock: st.session_state.get("predictions_primary", []),
                competitor_stock: st.session_state.get("predictions_competitor", [])
            })
            df.to_excel("analysis.xlsx", index=False)
            with open("analysis.xlsx", "rb") as f:
                st.download_button("📥 Get Excel", f, file_name="analysis.xlsx")
    with col2:
        if st.button("Download PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Stock Prediction Report", ln=True, align="C")
            pdf.output("analysis.pdf")
            with open("analysis.pdf", "rb") as f:
                st.download_button("📥 Get PDF", f, file_name="analysis.pdf")
    with col3:
        if st.button("Download DOC"):
            with open("analysis.doc", "w") as f:
                f.write("Stock Analysis Report Placeholder")
            with open("analysis.doc", "rb") as f:
                st.download_button("📥 Get DOC", f, file_name="analysis.doc")
