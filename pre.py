import os
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="📈 Stock Forecaster", layout="wide")
st.title("📈 LSTM & XGBoost Stock Forecasting App")

# --- Upload / Load Models ---
st.sidebar.header("🧠 Upload Models")
lstm_pep_model_file = st.sidebar.file_uploader("Upload PEP LSTM (.h5)", type="h5")
lstm_ko_model_file = st.sidebar.file_uploader("Upload KO LSTM (.h5)", type="h5")
xgb_pep_model_file = st.sidebar.file_uploader("Upload PEP XGBoost (.xgb)", type="xgb")
xgb_ko_model_file = st.sidebar.file_uploader("Upload KO XGBoost (.xgb)", type="xgb")

@st.cache_resource
def load_lstm_model(file):
    return load_model(file, compile=False)

@st.cache_resource
def load_xgb_model(file):
    model = xgb.Booster()
    model.load_model(file)
    return model

lstm_models = {}
if lstm_pep_model_file:
    lstm_models["PEP"] = load_lstm_model(lstm_pep_model_file)
    st.sidebar.success("PEP LSTM model loaded!")
if lstm_ko_model_file:
    lstm_models["KO"] = load_lstm_model(lstm_ko_model_file)
    st.sidebar.success("KO LSTM model loaded!")

xgb_models = {}
if xgb_pep_model_file:
    xgb_models["PEP"] = load_xgb_model(xgb_pep_model_file)
    st.sidebar.success("PEP XGBoost model loaded!")
if xgb_ko_model_file:
    xgb_models["KO"] = load_xgb_model(xgb_ko_model_file)
    st.sidebar.success("KO XGBoost model loaded!")

# --- Forecasting UI ---
st.subheader("📊 Forecast Stock Prices")
ticker = st.selectbox("Select Stock", ["PEP", "KO"])
model_type = st.radio("Model Type", ["LSTM"], horizontal=True)

if st.button("Predict Next 30 Days"):
    df = yf.download(ticker, period="5y", interval="1d")
    close = df[["Close"]].fillna(method='bfill')

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close)
    last_60 = scaled[-60:].reshape(1, 60, 1)

    if model_type == "LSTM" and ticker in lstm_models:
        model = lstm_models[ticker]
        predictions = []
        for _ in range(30):
            pred = model.predict(last_60, verbose=0)[0][0]
            predictions.append(pred)
            last_60 = np.append(last_60[:, 1:, :], [[[pred]]], axis=1)
        result = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        st.line_chart(pd.DataFrame(result, columns=["Forecasted Price"]))
    else:
        st.warning(f"Please upload the {model_type} model for {ticker}.")
