# Imports librairies
#from mlflow import MlflowClient
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np

# Set tracking experiment
mlflow.set_tracking_uri("http://127.0.0.1:8080")

# Define experiment name, run name and artifact_path name
apple_experiment = mlflow.set_experiment("Apple_Models")
run_name = "fourth_run"
artifact_path = "rf_apples"

# Import Database
data = pd.read_csv("data/fake_data.csv")
X = data.drop(columns=["date", "demand"])
X = X.astype('float')
y = data["demand"]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
params = {
    "n_estimators": 10,
    "max_depth": 15,
    "random_state": 42,
}
rf = RandomForestRegressor(**params)
rf.fit(X_train, y_train)

# Evaluate model
y_pred = rf.predict(X_val)
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, y_pred)
metrics = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}

# Store information in tracking server
with mlflow.start_run(run_name=run_name) as run:
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(
        sk_model=rf, input_example=X_val, artifact_path=artifact_path
    )
