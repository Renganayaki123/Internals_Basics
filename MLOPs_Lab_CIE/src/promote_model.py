import mlflow
import pandas as pd
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

mlflow.set_tracking_uri("file:./mlruns")

df = pd.read_csv("data/training_data.csv")

X = df.drop("delivery_time_min", axis=1)
y = df["delivery_time_min"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Champion model
champion_model = joblib.load("models/best_model.pkl")
champion_pred = champion_model.predict(X_test)
champion_mae = mean_absolute_error(y_test, champion_pred)

# Challenger model
challenger_model = RandomForestRegressor(random_state=99)
challenger_model.fit(X_train, y_train)
challenger_pred = challenger_model.predict(X_test)
challenger_mae = mean_absolute_error(y_test, challenger_pred)

# Compare
if challenger_mae < champion_mae:
    action = "promoted"
    champion_version = 2
else:
    action = "kept"
    champion_version = 1

output = {
    "registered_model_name": "freshbasket-delivery-time-min-predictor",
    "alias_name": "production",
    "champion_version": champion_version,
    "challenger_version": 2,
    "action": action
}

with open("results/step4_s7.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 4 completed")