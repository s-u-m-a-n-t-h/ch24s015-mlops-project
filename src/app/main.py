import streamlit as st
import requests
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Portfolio AI", layout="wide")

st.title("🚀 Portfolio AI: Intelligence & Optimization")

backend_url = os.getenv("BACKEND_URL", "http://backend:8000")

# --- Sidebar ---
st.sidebar.header("System Status")
if st.sidebar.button("Refresh Health"):
    try:
        response = requests.get(f"{backend_url}/health")
        if response.status_code == 200:
            health = response.json()
            st.sidebar.success(f"Backend: Online")
            st.sidebar.info(f"Model Loaded: {health.get('model_loaded')}")
        else:
            st.sidebar.error("Backend: Issues")
    except:
        st.sidebar.error("Backend: Offline")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Market Data", "🤖 Predictions", "⚖️ Optimization"])

# --- Tab 1: Market Data Visualization ---
with tab1:
    st.header("Historical Data & Indicators")
    
    col_t1, col_t2 = st.columns([1, 4])
    
    with col_t1:
        ticker = st.text_input("Ticker (e.g., AAPL)", value="AAPL")
        period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
        fetch_data = st.button("Fetch Market Data")
    
    if fetch_data or ticker:
        try:
            with st.spinner(f"Loading {ticker}..."):
                res = requests.post(f"{backend_url}/history", json={
                    "tickers": [ticker],
                    "period": period
                })
                res.raise_for_status()
                hist_data = res.json()[ticker]
                df = pd.DataFrame(hist_data)
                df['Date'] = pd.to_datetime(df['Date'])
                
                # Plotting
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                   vertical_spacing=0.1, subplot_titles=(f'{ticker} Price & BB', 'RSI'),
                                   row_width=[0.3, 0.7])

                # Candlestick / Price
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Adj Close'], name='Price', line=dict(color='blue')), row=1, col=1)
                fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Upper'], name='BB Upper', line=dict(dash='dash', color='gray')), row=1, col=1)
                fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Lower'], name='BB Lower', line=dict(dash='dash', color='gray')), row=1, col=1)
                
                # RSI
                fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

                fig.update_layout(height=600, template="plotly_dark", showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
                
                # Latest Indicators
                st.subheader("Latest Technical Metrics")
                latest = df.iloc[-1]
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric("RSI", f"{latest['RSI']:.2f}")
                m_col2.metric("MACD", f"{latest['MACD']:.2f}")
                m_col3.metric("BB Width", f"{(latest['BB_Upper'] - latest['BB_Lower']):.2f}")
                m_col4.metric("SMA 50", f"{latest['SMA_50']:.2f}")

        except Exception as e:
            st.error(f"Failed to fetch data: {e}")

# --- Tab 2: Predictions ---
with tab2:
    st.header("ML Model Prediction")
    st.write("Use the latest market data or input manual values to see the model's recommendation.")
    
    p_col1, p_col2 = st.columns(2)
    
    with p_col1:
        st.subheader("Features")
        # Initialize with dummy or latest data if available
        p_rsi = st.number_input("RSI Value", value=50.0)
        p_macd = st.number_input("MACD Value", value=0.0)
        p_macd_s = st.number_input("MACD Signal", value=0.0)
        p_bbu = st.number_input("BB Upper", value=100.0)
        p_bbl = st.number_input("BB Lower", value=90.0)
        p_sma20 = st.number_input("SMA 20", value=95.0)
        p_sma50 = st.number_input("SMA 50", value=94.0)

    with p_col2:
        st.subheader("Recommendation")
        if st.button("Run Model Inference", type="primary"):
            try:
                pred_res = requests.post(f"{backend_url}/predict", json={
                    "RSI": p_rsi, "MACD": p_macd, "MACD_Signal": p_macd_s,
                    "BB_Upper": p_bbu, "BB_Lower": p_bbl, 
                    "SMA_20": p_sma20, "SMA_50": p_sma50
                })
                pred_res.raise_for_status()
                prediction = pred_res.json()['prediction']
                
                if prediction == 1:
                    st.success("🎯 MODEL PREDICTION: **BUY / POSITIVE MOMENTUM**")
                else:
                    st.warning("⚠️ MODEL PREDICTION: **SELL / NEGATIVE MOMENTUM**")
            except Exception as e:
                st.error(f"Inference error: {e}")

# --- Tab 3: Optimization ---
with tab3:
    st.header("Portfolio Optimization (MVO)")
    st.write("Calculate optimal weights for a set of diversified assets.")
    
    risk_aversion = st.slider("Risk Aversion Level", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    
    if st.button("Optimize Weights", type="primary"):
        try:
            with st.spinner("Calculating optimal weights..."):
                opt_res = requests.post(f"{backend_url}/portfolio", json={
                    "risk_aversion": risk_aversion
                })
                opt_res.raise_for_status()
                allocation = opt_res.json()['allocation']
                
                # Visualize Allocation
                alloc_df = pd.DataFrame(list(allocation.items()), columns=['Asset', 'Weight'])
                alloc_df = alloc_df[alloc_df['Weight'] > 0.001] # Hide tiny weights
                
                pie_fig = go.Figure(data=[go.Pie(labels=alloc_df['Asset'], values=alloc_df['Weight'], hole=.3)])
                pie_fig.update_layout(template="plotly_dark", title="Optimal Asset Allocation")
                st.plotly_chart(pie_fig)
                
                st.table(alloc_df.sort_values(by='Weight', ascending=False))

        except Exception as e:
            st.error(f"Optimization error: {e}")

st.markdown("---")
st.caption("AI-Powered Portfolio Management System | Experimental v1.0")
