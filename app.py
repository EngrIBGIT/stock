import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from fpdf import FPDF
import streamlit as st


# --- Set Page Configuration ---
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

# --- Tabs for Navigation ---
tab1, tab2, tab3 = st.tabs(["Home", "Prediction", "Analysis"])

# --- Home Tab ---
with tab1:
    st.header("Welcome!")
    st.markdown(
        """
        This app provides stock price predictions, comparative analysis between primary and competitor stocks,
        and detailed insights using benchmark indices. 🌟 Select models, input variables, and enjoy visualizations
        of market trends!
        """
    )

# --- Prediction Tab ---
with tab2:
    st.header("Prediction Models")

    # Dropdown for stock selection
    primary_stock = st.selectbox("Select Primary Stock (e.g., PEP):", ["PEP", "AAPL", "GOOG", "MSFT"], key="primary_stock")
    competitor_stock = st.selectbox("Select Competitor Stock (e.g., KO):", ["KO", "TSLA", "META", "AMZN"], key="competitor_stock")
    indices = st.selectbox("Select Benchmark Index:", ["S&P 500", "NASDAQ", "Dow Jones"], key="benchmark_index")

    # Choose model
    model_choice = st.radio("Choose Prediction Model:", ["XGBoost", "LSTM"], key="model_choice")

    # Input features with unique keys
    st.subheader("Enter Input Variables (Best Performing Features):")
    input_features = {}
    for feature in [
        'Volume', 'Adj Close', 'Daily Return', 'Tomorrow', 'Cumulative Return',
        'SMA_20', 'EMA_20', 'RSI', 'MACD', 'Signal_Line', 'Volatility',
        'Middle_Band', 'Std_Dev'
    ]:
        input_features[feature] = st.number_input(f"{feature} Value:", 0.0, 100.0, step=0.1, key=f"input_{feature}")

    # Button to trigger prediction with a unique key
    if st.button("Predict Stock Prices", key="predict_button"):
        if model_choice == "XGBoost":
            st.write("Using XGBoost...")
            # Mock predictions for demonstration (Replace with actual model implementation)
            predictions_primary = np.random.uniform(100, 200, 5)
            predictions_competitor = np.random.uniform(80, 180, 5)
        elif model_choice == "LSTM":
            st.write("Using LSTM...")
            predictions_primary = np.random.uniform(120, 220, 5)
            predictions_competitor = np.random.uniform(90, 190, 5)

        # Display predictions
        st.write(f"{primary_stock} Predicted Prices:", predictions_primary)
        st.write(f"{competitor_stock} Predicted Prices:", predictions_competitor)

# --- Analysis Tab ---
with tab3:
    st.header("Comparative Analysis")

    # Example visualizations (Replace with real data)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.random.uniform(100, 200, 10), label=f"{primary_stock} Trends")
    ax.plot(np.random.uniform(90, 190, 10), label=f"{competitor_stock} Trends")
    ax.legend()
    st.pyplot(fig)

    # Download options
    st.subheader("Download Analysis")
    if st.button("Download Excel"):
        st.write("Downloading Excel...")
        # Mock file creation for demonstration
        with open("analysis.xlsx", "w") as f:
            f.write("Analysis Data")
    if st.button("Download PDF"):
        st.write("Downloading PDF...")
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Stock Analysis Report", ln=True, align="C")
        pdf.output("analysis.pdf")
    if st.button("Download Document"):
        st.write("Downloading Document...")
        with open("analysis.doc", "w") as f:
            f.write("Analysis Data")