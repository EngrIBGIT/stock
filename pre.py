import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="LSTM Overfitting Check", layout="wide")
st.title("📊 LSTM Stock Forecasting with Overfitting Check")

# --- Step 1: Stock Selection ---
stock_choice = st.selectbox("Choose Stock to Train", ["PEP", "KO"])
ticker = "PEP" if stock_choice == "PEP" else "KO"

# --- Step 2: Data Download ---
@st.cache_data
def get_data(ticker):
    df = yf.download(ticker, start="2020-01-01", end="2024-12-02")
    df = df[['Close']].copy()
    df.dropna(inplace=True)
    df['Adj Close'] = df['Close']
    return df

df = get_data(ticker)

# --- Step 3: Preprocess ---
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[['Adj Close']])

x, y = [], []
for i in range(60, len(scaled)):
    x.append(scaled[i-60:i, 0])
    y.append(scaled[i, 0])

x = np.array(x)
y = np.array(y)
x = np.reshape(x, (x.shape[0], x.shape[1], 1))

# --- Step 4: Train/Val Split ---
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# --- Step 5: Train Model with Validation ---
def build_lstm(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

model = build_lstm((x_train.shape[1], 1))

with st.spinner("Training LSTM model..."):
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=10,
        batch_size=1,
        verbose=0
    )

# --- Step 6: Plot Loss ---
st.subheader(f"📉 {stock_choice} Training vs Validation Loss")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(history.history['loss'], label='Train Loss')
ax.plot(history.history['val_loss'], label='Val Loss')
ax.set_title(f"{stock_choice} LSTM: Training vs Validation Loss")
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# --- Advice ---
st.info("👉 If validation loss diverges from training loss, your model is likely overfitting.")

