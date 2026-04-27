import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import joblib

def train_model(input_dir):
    """
    Trains multiple models with different hyperparameters and logs to MLflow.
    """
    files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    if not files:
        print(f"No feature files found in {input_dir}")
        return

    # Combine all feature files for training
    all_data = []
    for file in files:
        df = pd.read_csv(os.path.join(input_dir, file))
        all_data.append(df)
    
    data = pd.concat(all_data, ignore_index=True)
    
    # Define features and target
    features = ['RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'SMA_20', 'SMA_50']
    X = data[features]
    y = data['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # MLflow Setup
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("Portfolio_Allocation_Experiment")
    
    # Hyperparameter Grid
    n_estimators_list = [50, 100, 200]
    max_depth_list = [5, 10, None]
    
    best_f1 = 0
    best_run_id = None
    
    for n_estimators in n_estimators_list:
        for max_depth in max_depth_list:
            with mlflow.start_run(run_name=f"RF_n{n_estimators}_d{max_depth}"):
                # Initialize and train model
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                model.fit(X_train, y_train)
                
                # Predictions and metrics
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred)
                rec = recall_score(y_test, y_pred)
                
                # Log parameters and metrics
                mlflow.log_param("n_estimators", n_estimators)
                mlflow.log_param("max_depth", max_depth)
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("precision", prec)
                mlflow.log_metric("recall", rec)
                
                # Log model
                mlflow.sklearn.log_model(model, "model")
                
                print(f"Run completed: n_estimators={n_estimators}, max_depth={max_depth}, F1={f1:.4f}")
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_run_id = mlflow.active_run().info.run_id

    # Register the best model
    if best_run_id:
        model_name = "portfolio_allocation_model"
        model_uri = f"runs:/{best_run_id}/model"
        mv = mlflow.register_model(model_uri, model_name)
        
        # Transition to Production
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage="Production"
        )
        print(f"Best model registered and transitioned to Production: {model_name} v{mv.version}")

if __name__ == "__main__":
    features_path = "/opt/airflow/data/features"
    train_model(features_path)
