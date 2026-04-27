from fastapi import FastAPI, HTTPException
import mlflow.sklearn
import pandas as pd
import os
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Portfolio Allocation API")

class PredictionInput(BaseModel):
    RSI: float
    MACD: float
    MACD_Signal: float
    BB_Upper: float
    BB_Lower: float
    SMA_20: float
    SMA_50: float

class PredictionOutput(BaseModel):
    prediction: int

model_name = "portfolio_allocation_model"
model_stage = "Production"
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)

def load_model():
    try:
        model_uri = f"models:/{model_name}/{model_stage}"
        model = mlflow.sklearn.load_model(model_uri)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

@app.get("/")
def read_root():
    return {"message": "Portfolio Allocation API is running"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    model = load_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    # Convert input data to DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Predict
    prediction = model.predict(input_df)[0]
    
    return {"prediction": int(prediction)}
