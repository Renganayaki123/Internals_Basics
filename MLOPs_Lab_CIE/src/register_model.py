import mlflow
import mlflow.sklearn
import joblib
import json

mlflow.set_tracking_uri("file:./mlruns")

model = joblib.load("models/best_model.pkl")

with mlflow.start_run() as run:

    result = mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="freshbasket-delivery-time-min-predictor"
    )

    run_id = run.info.run_id

output = {
    "registered_model_name": "freshbasket-delivery-time-min-predictor",
    "version": 1,
    "run_id": run_id,
    "source_metric": "mae",
    "source_metric_value": 0.0
}

with open("results/step3_s6.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 3 completed")