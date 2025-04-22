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

# --- Load Models Automatically ---
@st.cache_resource
def load_lstm_model(path):
    return load_model(path, compile=False)

@st.cache_resource
def load_xgb_model(path):
    model = xgb.Booster()
    model.load_model(path)
    return model

# Load models from local files
lstm_models = {
    "PEP": load_lstm_model("pep_lstm_model.h5") if os.path.exists("pep_lstm_model.h5") else None,
    "KO": load_lstm_model("ko_lstm_model.h5") if os.path.exists("ko_lstm_model.h5") else None
}

xgb_models = {
    "PEP": load_xgb_model("xgb_pep_model.xgb") if os.path.exists("xgb_pep_model.xgb") else None,
    "KO": load_xgb_model("xgb_ko_model.xgb") if os.path.exists("xgb_ko_model.xgb") else None
}

# --- Forecast UI ---
st.subheader("📊 Forecast Stock Prices")
ticker = st.selectbox("Select Stock", ["PEP", "KO"])
model_type = st.radio("Model Type", ["LSTM", "XGBoost"], horizontal=True)

# --- Feature Input Section for XGBoost ---
st.markdown("### 🔢 Input Features for XGBoost")
xgb_inputs = {}
xgb_features = ['Adj Close', 'Volume', 'RSI', 'SMA_20', 'MACD']
for feat in xgb_features:
    xgb_inputs[feat] = st.number_input(f"{feat}:", value=50.0, key=feat)

# --- Prediction Trigger ---
if st.button("🔮 Predict Next 30 Days"):
    try:
        df = yf.download(ticker, period="5y", interval="1d")
        df = df[["Close"]].dropna()
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df)
        last_60 = scaled[-60:].reshape(1, 60, 1)

        if model_type == "LSTM":
            if lstm_models[ticker] is None:
                st.error(f"⚠️ LSTM model for {ticker} not found.")
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

        elif model_type == "XGBoost":
            if xgb_models[ticker] is None:
                st.error(f"⚠️ XGBoost model for {ticker} not found.")
            else:
                x_input = np.array([list(xgb_inputs.values())])
                dmatrix = xgb.DMatrix(x_input, feature_names=xgb_features)
                prediction = xgb_models[ticker].predict(dmatrix)
                st.success(f"📈 {ticker} Forecasted Value: {prediction[0]:.2f}")

    except Exception as e:
        st.error(f"An error occurred during forecasting: {e}")
