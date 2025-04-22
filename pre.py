import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler

# --- App Config ---
st.set_page_config(page_title="📈 Stock Forecaster", layout="wide")
st.title("📈 LSTM & XGBoost Stock Forecasting App")

# --- Sidebar: Upload Models ---
st.sidebar.header("🧠 Upload Trained Models")
lstm_pep_model_file = st.sidebar.file_uploader("Upload PEP LSTM (.h5)", type="h5")
lstm_ko_model_file = st.sidebar.file_uploader("Upload KO LSTM (.h5)", type="h5")
xgb_pep_model_file = st.sidebar.file_uploader("Upload PEP XGBoost (.json)", type="json")
xgb_ko_model_file = st.sidebar.file_uploader("Upload KO XGBoost (.json)", type="json")

# --- Model Loaders with Error Handling ---
@st.cache_resource
def load_lstm_model(file):
    try:
        return load_model(file, compile=False)
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load LSTM model: {e}")
        return None

@st.cache_resource
def load_xgb_model(file):
    try:
        model = xgb.Booster()
        model.load_model(file)
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load XGBoost model: {e}")
        return None

# --- Load Uploaded Models ---
lstm_models = {}
if lstm_pep_model_file:
    lstm_models["PEP"] = load_lstm_model(lstm_pep_model_file)
    if lstm_models["PEP"]: st.sidebar.success("✅ PEP LSTM model loaded")

if lstm_ko_model_file:
    lstm_models["KO"] = load_lstm_model(lstm_ko_model_file)
    if lstm_models["KO"]: st.sidebar.success("✅ KO LSTM model loaded")

xgb_models = {}
if xgb_pep_model_file:
    xgb_models["PEP"] = load_xgb_model(xgb_pep_model_file)
    if xgb_models["PEP"]: st.sidebar.success("✅ PEP XGBoost model loaded")

if xgb_ko_model_file:
    xgb_models["KO"] = load_xgb_model(xgb_ko_model_file)
    if xgb_models["KO"]: st.sidebar.success("✅ KO XGBoost model loaded")

# --- Forecast Section ---
st.subheader("📊 Forecast Stock Prices")
ticker = st.selectbox("Select Stock", ["PEP", "KO"])
model_type = st.radio("Model Type", ["LSTM"], horizontal=True)

# --- Forecast Logic ---
if st.button("🔮 Predict Next 30 Days"):
    try:
        df = yf.download(ticker, period="5y", interval="1d")
        df = df[["Close"]].dropna()
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df)

        last_60 = scaled[-60:].reshape(1, 60, 1)

        if model_type == "LSTM":
            if ticker not in lstm_models or lstm_models[ticker] is None:
                st.error(f"⚠️ Please upload a valid LSTM model for {ticker}.")
            else:
                model = lstm_models[ticker]
                predictions = []
                for _ in range(30):
                    pred = model.predict(last_60, verbose=0)[0][0]
                    predictions.append(pred)
                    last_60 = np.append(last_60[:, 1:, :], [[[pred]]], axis=1)

                forecasted = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
                forecast_df = pd.DataFrame(forecasted, columns=["Forecasted Price"])
                st.success(f"✅ Forecast for {ticker} complete.")
                st.line_chart(forecast_df)
    except Exception as e:
        st.error(f"An error occurred during forecasting: {e}")
