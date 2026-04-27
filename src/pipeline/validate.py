import pandas as pd
import os
import sys

def validate_data(input_dir):
    """
    Validates the raw market data files.
    """
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist.")
        sys.exit(1)

    files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    if not files:
        print(f"Error: No CSV files found in {input_dir}.")
        sys.exit(1)

    for file in files:
        file_path = os.path.join(input_dir, file)
        print(f"Validating {file}...")
        
        try:
            df = pd.read_csv(file_path)
            
            # 1. Basic Schema Check
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                else:
                    print(f"Warning: Missing column {col} in {file}")

            # 2. Check for empty files
            if df.empty:
                print(f"Error: {file} is empty.")
                sys.exit(1)

            # 3. Check for NaNs
            nan_count = df.isnull().sum().sum()
            if nan_count > 0:
                print(f"Warning: {file} contains {nan_count} missing values. Handling them...")
                # Simple forward fill for time series
                df.fillna(method='ffill', inplace=True)
                df.to_csv(file_path, index=False)

        except Exception as e:
            print(f"Error validating {file}: {e}")
            sys.exit(1)

    print("Data validation successful.")

if __name__ == "__main__":
    # Path relative to Airflow container root
    raw_data_path = "/opt/airflow/data/raw"
    validate_data(raw_data_path)
