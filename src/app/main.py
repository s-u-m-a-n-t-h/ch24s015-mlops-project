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
            st.sidebar.error("API is unreachable")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

st.write("Welcome to the Portfolio Allocation Dashboard. Select your parameters to get started.")
