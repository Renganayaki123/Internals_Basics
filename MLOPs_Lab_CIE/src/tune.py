import pandas as pd
import json
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("freshbasket-delivery-time-min")

df = pd.read_csv("data/training_data.csv")

X = df.drop("delivery_time_min", axis=1)
y = df["delivery_time_min"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

params = {
    "n_estimators": [50, 150, 250],
    "max_depth": [5, 10, 20],
    "min_samples_split": [2, 3, 5]
}

with mlflow.start_run(run_name="tuning-freshbasket"):

    model = RandomForestRegressor(random_state=42)

    grid = GridSearchCV(
        model,
        params,
        cv=5,
        scoring="neg_mean_absolute_error"
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    pred = best_model.predict(X_test)

    best_mae = mean_absolute_error(y_test, pred)

output = {
    "search_type": "grid",
    "n_folds": 5,
    "total_trials": 27,
    "best_params": grid.best_params_,
    "best_mae": float(best_mae),
    "best_cv_mae": float(-grid.best_score_),
    "parent_run_name": "tuning-freshbasket"
}

with open("results/step2_s2.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 2 completed")