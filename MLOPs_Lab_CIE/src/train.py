import pandas as pd
import json
import os
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import numpy as np

os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("freshbasket-delivery-time-min")

df = pd.read_csv("data/training_data.csv")

X = df.drop("delivery_time_min", axis=1)
y = df["delivery_time_min"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Ridge": Ridge(),
    "RandomForest": RandomForestRegressor(random_state=42)
}

results = []
best_model = None
best_mae = float("inf")
best_name = ""

for name, model in models.items():

    with mlflow.start_run(run_name=name):

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))

        mlflow.log_param("model_name", name)

        if name == "RandomForest":
            mlflow.log_param("random_state", 42)

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.set_tag("priority", "high")

        mlflow.sklearn.log_model(model, name)

        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_name = name
            joblib.dump(model, "models/best_model.pkl")

        results.append({
            "name": name,
            "mae": float(mae),
            "rmse": float(rmse)
        })

output = {
    "experiment_name": "freshbasket-delivery-time-min",
    "models": results,
    "best_model": best_name,
    "best_metric_name": "mae",
    "best_metric_value": float(best_mae)
}

with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 1 completed")