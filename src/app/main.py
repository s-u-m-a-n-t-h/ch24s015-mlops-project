import streamlit as st
import requests
import os

st.set_page_config(page_title="Portfolio Allocation", layout="wide")

st.title("Automated Portfolio Allocation System")

backend_url = os.getenv("BACKEND_URL", "http://backend:8000")

st.sidebar.header("Settings")
if st.sidebar.button("Check API Health"):
    try:
        response = requests.get(f"{backend_url}/health")
        if response.status_code == 200:
            st.sidebar.success("API is healthy")
        else:
            st.sidebar.error(f"API returned status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"API connection error: {e}")

st.header("Get Prediction")

# Input fields for prediction
st.write("Enter the following technical indicators to get a prediction:")

col1, col2, col3 = st.columns(3)

with col1:
    rsi = st.number_input("RSI", value=55.5, step=0.1)
    bb_upper = st.number_input("Bollinger Band Upper", value=150.0, step=0.1)
    sma_20 = st.number_input("SMA 20", value=145.0, step=0.1)

with col2:
    macd = st.number_input("MACD", value=1.2, step=0.1)
    bb_lower = st.number_input("Bollinger Band Lower", value=140.0, step=0.1)
    sma_50 = st.number_input("SMA 50", value=142.0, step=0.1)

with col3:
    macd_signal = st.number_input("MACD Signal", value=0.8, step=0.1)
    # Placeholder for potential future inputs
    st.write("") # Empty for spacing
    st.write("") # Empty for spacing

predict_button = st.button("Get Prediction")

if predict_button:
    prediction_input = {
        "RSI": rsi,
        "MACD": macd,
        "MACD_Signal": macd_signal,
        "BB_Upper": bb_upper,
        "BB_Lower": bb_lower,
        "SMA_20": sma_20,
        "SMA_50": sma_50,
    }

    try:
        with st.spinner("Getting prediction..."):
            response = requests.post(f"{backend_url}/predict", json=prediction_input)
            response.raise_for_status() # Raise an exception for bad status codes
            prediction = response.json()
            
            st.subheader("Prediction Result:")
            st.write(f"The model recommends: **{prediction.get('prediction', 'N/A')}**")
            
            # Display raw JSON response for debugging if needed
            # st.json(prediction)

    except requests.exceptions.RequestException as e:
        st.error(f"Error getting prediction: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

st.markdown("---")
st.write("Note: This is a simplified interface. Future enhancements will include historical data visualization and more advanced parameter controls.")
