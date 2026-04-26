import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

def fetch_market_data(tickers, period="2y", interval="1d"):
    """
    Downloads historical data for a list of tickers.
    """
    data_dir = "data/raw"
    os.makedirs(data_dir, exist_ok=True)
    
    metadata = []
    
    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        df = yf.download(ticker, period=period, interval=interval)
        
        if not df.empty:
            file_path = os.path.join(data_dir, f"{ticker}.csv")
            df.to_csv(file_path)
            metadata.append({
                "ticker": ticker,
                "file_path": file_path,
                "rows": len(df),
                "last_fetched": datetime.now().isoformat()
            })
            
    return pd.DataFrame(metadata)

if __name__ == "__main__":
    # Default basket of stocks for the portfolio
    default_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    results = fetch_market_data(default_tickers)
    print("\nIngestion Summary:")
    print(results)
