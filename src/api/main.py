from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import os
import mlflow

# --- MLflow Configuration ---
# Ensure MLflow tracking URI and artifact root are correctly set from environment variables
# These should be configured in docker-compose.yml for the backend service.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
# The artifact root is typically set by the MLflow server's configuration,
# but if the client needs to know where to save temporary artifacts before logging,
# it might be useful. However, for loading models, it's more critical.
# MLFLOW_ARTIFACT_ROOT = os.getenv("MLFLOW_ARTIFACT_ROOT", "/opt/mlflow/artifacts") # This might not be directly used by client here

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
print(f"MLflow Tracking URI set to: {MLFLOW_TRACKING_URI}")

# --- Model Loading ---
# Load the production model from MLflow
# The model is expected to be registered as 'portfolio_allocation_model' and in 'Production' stage.
MODEL_NAME = "portfolio_allocation_model"
try:
    # Find the latest production model version
    client = mlflow.MlflowClient()
    model_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    prod_model_version = None
    for mv in model_versions:
        if mv.current_stage == "Production":
            prod_model_version = mv
            break

    if prod_model_version:
        model_uri = f"models:/{MODEL_NAME}/Production"
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        print(f"Successfully loaded model '{MODEL_NAME}' version {prod_model_version.version} from MLflow.")
    else:
        loaded_model = None
        print(f"No production version found for model '{MODEL_NAME}'. Model serving will be unavailable.")
        # Optionally, load the latest staging model or a fallback
        # For now, we'll proceed without a loaded model if Production is not found.

except Exception as e:
    loaded_model = None
    print(f"Error loading MLflow model '{MODEL_NAME}': {e}")
    # In a real application, you might want to log this error more formally or have a fallback.

# --- Pydantic Models ---
class PredictionInput(BaseModel):
    RSI: float
    MACD: float
    MACD_Signal: float
    BB_Upper: float
    BB_Lower: float
    SMA_20: float
    SMA_50: float

class PortfolioInput(BaseModel):
    risk_aversion: float = 1.0 # Higher value means more risk-averse

class PredictionOutput(BaseModel):
    prediction: int # Assuming 0 or 1 based on previous curl response

class PortfolioAllocationOutput(BaseModel):
    allocation: dict # e.g., {"AAPL": 0.4, "GOOG": 0.3, "MSFT": 0.3}

# --- FastAPI App ---
app = FastAPI()

# Add a health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "API is healthy", "model_loaded": loaded_model is not None}

# Prediction Endpoint
@app.post("/predict", response_model=PredictionOutput)
async def predict(data: PredictionInput):
    if loaded_model is None:
        raise HTTPException(status_code=503, detail="Model not available") # Use 503 Service Unavailable

    # Prepare input data for the model (this needs to match how the model expects input)
    # Assuming the model expects a pandas DataFrame with specific columns
    # This part is crucial and depends on how train.py prepared the data/model
    # For now, creating a dummy DataFrame. You'll need to adjust column names and order.
    try:
        # IMPORTANT: Adjust these column names to EXACTLY match what your model expects
        # It's best if train.py logs the feature names or if they are hardcoded based on training.
        # Based on input fields: RSI, MACD, MACD_Signal, BB_Upper, BB_Lower, SMA_20, SMA_50
        # Let's assume the model expects them in a certain order or with specific names.
        # For demonstration, creating a DataFrame. If your model expects a specific order or names, adjust here.
        input_df = pd.DataFrame([{
            "RSI": data.RSI,
            "MACD": data.MACD,
            "MACD_Signal": data.MACD_Signal,
            "BB_Upper": data.BB_Upper,
            "BB_Lower": data.BB_Lower,
            "SMA_20": data.SMA_20,
            "SMA_50": data.SMA_50,
        }])

        # Make prediction
        # The model output needs to be processed to match PredictionOutput.prediction (int)
        prediction_result = loaded_model.predict(input_df) # This returns an array, e.g., [0] or [1]
        
        # Ensure the output is an integer as per PredictionOutput model
        predicted_value = int(prediction_result[0]) if isinstance(prediction_result, (list, np.ndarray)) and len(prediction_result) > 0 else int(prediction_result)

        return PredictionOutput(prediction=predicted_value)
    except Exception as e:
        print(f"Error during prediction: {e}") # Log the error server-side
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


# --- Portfolio Optimization Logic ---

# Function to calculate historical data stats
def get_historical_data_stats(tickers: list[str], period: str = "2y", interval: str = "1d"):
    """
    Fetches historical data for given tickers and calculates:
    - Expected annual returns
    - Annualized volatility
    - Annualized covariance matrix
    """
    try:
        # Fetch data
        data = yf.download(tickers, period=period, interval=interval, auto_adjust=True)['Adj Close']
        if data.empty:
            raise ValueError("No data fetched from yfinance.")
            
        # Calculate daily returns
        returns = data.pct_change().dropna()
        
        # Calculate annualized metrics
        # Assuming 252 trading days in a year
        num_trading_days = 252 
        
        expected_returns = returns.mean() * num_trading_days
        cov_matrix = returns.cov() * num_trading_days
        
        return expected_returns, cov_matrix
        
    except Exception as e:
        print(f"Error fetching or processing historical data: {e}")
        # Fallback: If yfinance fails or data is insufficient, we can use placeholders or return an error.
        # For demonstration, let's create dummy data if fetching fails.
        # In a real app, you'd want robust error handling or pre-computed data.
        if 'tickers' in locals() and tickers:
            print(f"Creating dummy stats for tickers: {tickers}")
            # Create dummy returns and covariance matrix
            dummy_returns = pd.DataFrame(np.random.randn(100, len(tickers)) * 0.01, columns=tickers)
            dummy_expected_returns = dummy_returns.mean() * num_trading_days
            dummy_cov_matrix = dummy_returns.cov() * num_trading_days
            return dummy_expected_returns, dummy_cov_matrix
        else:
            raise e # Re-raise if tickers is empty or no other fallback is possible

# Portfolio optimization function using Mean-Variance Optimization (MVO)
def portfolio_performance(weights, cov_matrix):
    """Calculates portfolio volatility (standard deviation)."""
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def neg_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate=0.0):
    """Calculates the negative Sharpe Ratio for a portfolio."""
    p_returns = np.dot(weights, expected_returns)
    p_volatility = portfolio_performance(weights, cov_matrix)
    if p_volatility == 0: # Avoid division by zero
        return 0 # Or some other indicator of invalidity
    sharpe_ratio = (p_returns - risk_free_rate) / p_volatility
    return -sharpe_ratio # Minimize negative Sharpe Ratio

def minimize_volatility(weights, expected_returns, cov_matrix, target_return, risk_free_rate=0.0):
    """Calculates portfolio variance for a target return, constrained by weights sum to 1."""
    # This function is to minimize variance for a TARGET return
    # It's more direct to optimize for max Sharpe, but this is also common.
    # Let's stick to maximizing Sharpe ratio for simplicity unless target return is critical.
    # For now, implementing a function that minimizes variance subject to target return constraint.
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_volatility = portfolio_performance(weights, cov_matrix)
    
    # We want to minimize variance, so the objective is portfolio_volatility,
    # but it's implicitly tied to the target_return constraint.
    # If we were to minimize variance directly, we'd just return portfolio_volatility.
    # For MVO, typically we optimize for max Sharpe or min variance for a given return.
    # Let's reformulate to minimize variance, and use target_return as a constraint.
    return portfolio_volatility # Objective: minimize volatility


def optimize_portfolio(expected_returns, cov_matrix, risk_aversion=1.0, risk_free_rate=0.0):
    """
    Finds optimal portfolio weights using Mean-Variance Optimization.
    Optimizes for maximum Sharpe Ratio.
    """
    num_assets = len(expected_returns)
    args = (expected_returns, cov_matrix, risk_free_rate)

    # Constraints: sum of weights is 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    # Bounds: weights are between 0 and 1 (no short selling)
    bounds = tuple((0.0, 1.0) for asset in range(num_assets))

    # Initial guess: equal weights
    initial_weights = num_assets * [1. / num_assets,]

    # Optimize for maximum Sharpe Ratio
    # We minimize the negative Sharpe Ratio
    # Note: For risk_aversion, a higher value could imply higher risk tolerance,
    # leading to higher expected returns and potentially higher risk.
    # A common way to incorporate risk aversion is to scale the expected returns by (1/risk_aversion)
    # or use it to set a target return.
    # For simplicity here, we'll focus on maximizing Sharpe ratio.
    # If risk_aversion is explicitly passed and we want to use it to scale returns:
    # adjusted_expected_returns = expected_returns / risk_aversion # Higher risk_aversion -> lower expected return

    # Let's implement maximization of Sharpe ratio as the primary optimization goal.
    # The `risk_aversion` parameter can be used to adjust target return or as a scaling factor.
    # For now, we'll just use it to potentially scale the target return if we were to implement that.
    # Let's redefine to use risk aversion for maximizing sharpe ratio, but it needs a bit more context.
    # A simpler approach is to optimize for max Sharpe Ratio directly.

    # We'll optimize for maximum Sharpe ratio. The risk_aversion can be used to bias towards higher returns.
    # A common approach for MVO with risk aversion is: Maximize (Expected Portfolio Return - RiskFreeRate) - 0.5 * RiskAversion * PortfolioVariance
    # This is equivalent to maximizing E[Rp] - lambda * Var[Rp] where lambda is risk aversion coefficient.

    # Let's re-implement the objective function to include risk aversion
    # Objective: Maximize Sharpe Ratio = (Rp - Rf) / Sigma_p
    # Equivalent to Minimize -(Rp - Rf) / Sigma_p
    # Or, using risk aversion directly: Maximize E[Rp] - lambda * Var[Rp]
    # Minimize -(E[Rp] - lambda * Var[Rp])

    # Objective function to minimize: - (portfolio_return - risk_free_rate) + risk_aversion * portfolio_volatility**2
    # This is a common form to incorporate risk aversion.
    def objective_function(weights, expected_returns, cov_matrix, risk_free_rate, risk_aversion):
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_volatility_sq = np.dot(weights.T, np.dot(cov_matrix, weights)) # Variance
        # We want to maximize return and minimize risk (variance). The risk_aversion parameter scales the penalty for variance.
        # Higher risk_aversion means higher penalty for variance, thus favoring lower risk.
        return -(portfolio_return - risk_free_rate) + risk_aversion * portfolio_volatility_sq

    # Optimization
    result = minimize(
        objective_function,
        initial_weights,
        args=(expected_returns, cov_matrix, risk_free_rate, risk_aversion),
        method='SLSQP', # Sequential Least Squares Programming, good for constrained optimization
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        print(f"Portfolio optimization failed: {result.message}")
        # Fallback: return equal weights or raise an error
        return {expected_returns.index[i]: weight for i, weight in enumerate(initial_weights)}
    
    return {expected_returns.index[i]: weight for i, weight in enumerate(result.x)}

# --- API Endpoint for Portfolio Allocation ---
@app.post("/portfolio", response_model=PortfolioAllocationOutput)
async def get_portfolio_allocation(portfolio_input: PortfolioInput):
    """
    Calculates and returns the optimal portfolio allocation based on risk aversion.
    """
    # Fetch data and calculate stats
    # You might want to cache this or fetch from a more robust source than yfinance on every call.
    # For demonstration, we'll use a fixed period and interval.
    # It's good practice to get tickers from configuration or a known list.
    # Let's assume we are optimizing for a fixed set of common tickers.
    # If you want to optimize for tickers related to your prediction model, that's a different flow.
    
    # For now, let's use a hardcoded list of tickers commonly available.
    # In a real scenario, this list should be dynamic or configurable.
    OPTIMIZATION_TICKERS = ['AAPL', 'GOOG', 'MSFT', 'AMZN', 'META', 'NVDA'] # Example tickers
    
    try:
        expected_returns, cov_matrix = get_historical_data_stats(OPTIMIZATION_TICKERS, period="2y", interval="1d")
        
        # Ensure all tickers are present in both returns and cov_matrix
        if (not all(ticker in expected_returns.index for ticker in OPTIMIZATION_TICKERS) or 
            not all(ticker in cov_matrix.columns for ticker in OPTIMIZATION_TICKERS)):
            raise ValueError("Mismatch between tickers and calculated stats.")

        # Align expected_returns and cov_matrix to the OPTIMIZATION_TICKERS order for optimization
        expected_returns_ordered = expected_returns[OPTIMIZATION_TICKERS]
        cov_matrix_ordered = cov_matrix.loc[OPTIMIZATION_TICKERS, OPTIMIZATION_TICKERS]
        
        allocation = optimize_portfolio(
            expected_returns_ordered,
            cov_matrix_ordered,
            risk_aversion=portfolio_input.risk_aversion
        )
        
        return PortfolioAllocationOutput(allocation=allocation)

    except ValueError as ve:
        print(f"Configuration error: {ve}")
        raise HTTPException(status_code=400, detail=f"Configuration error: {ve}")
    except Exception as e:
        print(f"Error calculating portfolio allocation: {e}")
        raise HTTPException(status_code=500, detail=f"Portfolio calculation failed: {e}")

# --- Example usage for testing (optional, can be removed for production) ---
if __name__ == "__main__":
    import uvicorn
    # To run this locally for testing:
    # 1. Make sure you have run `pip install -r requirements.txt` (including scipy)
    # 2. Set MLFLOW_TRACKING_URI=http://localhost:5000 (if running MLflow locally)
    # 3. Run `uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000`
    #    or use `docker compose up` after rebuilding the backend.
    # Note: The data loading will rely on local files or network access to yfinance.
    
    # For testing purposes without actual MLflow model loading, uncomment below:
    # print("Running in test mode without MLflow model loading.")
    # loaded_model = "dummy_model" # Simulate a loaded model
    # @app.post("/predict", response_model=PredictionOutput)
    # async def predict_test(data: PredictionInput):
    #     return PredictionOutput(prediction=0) # Dummy prediction
        
    print("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
