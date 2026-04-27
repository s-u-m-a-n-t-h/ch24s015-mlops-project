import pandas as pd
import numpy as np
import os
import sys

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, slow=26, fast=12, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_bollinger_bands(series, period=20, std_dev=2):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, lower

def engineer_features(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    
    for file in files:
        print(f"Engineering features for {file} (Pure Pandas)...")
        df = pd.read_csv(os.path.join(input_dir, file))
        
        # Ensure numeric types for calculation
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.sort_values('Date', inplace=True)
        
        # 1. Technical Indicators
        df['RSI'] = calculate_rsi(df['Close'])
        df['MACD'], df['MACD_Signal'] = calculate_macd(df['Close'])
        df['BB_Upper'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # 2. Target Variable: 1 if 5-day return > 0, else 0
        df['Target_Next_5d_Close'] = df['Close'].shift(-5)
        df['Target'] = (df['Target_Next_5d_Close'] > df['Close']).astype(int)
        
        df.dropna(inplace=True)
        
        output_path = os.path.join(output_dir, file)
        df.to_csv(output_path, index=False)
        print(f"Features saved to {output_path} (Rows: {len(df)})")

if __name__ == "__main__":
    raw_path = "/opt/airflow/data/raw"
    features_path = "/opt/airflow/data/features"
    engineer_features(raw_path, features_path)
