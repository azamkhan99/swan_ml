# train_ngboost.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import MLE
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
import mlflow

from utils import (
    create_lagged_and_cumulative_features,
    nse_score,
    pbias_score,
    r_squared,
    select_features,
)

# Set paths
data_path = "preprocessed_data/"
os.makedirs("models", exist_ok=True)

targets = [
    "Phosphate DlyLd(kg)",
    "Nitrate DlyLd(kg)",
    "Sediments DlyLd(kg*1000)",
]

features = {
    "Sediments DlyLd(kg*1000)": [
        "Calibrated_SWAT_Streamflow",
        "Calibrated_SWAT_Nitrate DlyLd(kg)",
        "Calibrated_SWAT_Phosphate DlyLd(kg)",
    ],
    "Nitrate DlyLd(kg)": [
        "Calibrated_SWAT_Streamflow",
        "Calibrated_SWAT_Sediments DlyLd(kg*1000)",
        "Calibrated_SWAT_Phosphate DlyLd(kg)",
    ],
    "Phosphate DlyLd(kg)": [
        "Calibrated_SWAT_Streamflow",
        "Calibrated_SWAT_Sediments DlyLd(kg*1000)",
        "Calibrated_SWAT_Nitrate DlyLd(kg)",
    ],
}


def hyperopt_ngboost(X_train, y_train, X_val, y_val):
    def objective(params):
        model = NGBRegressor(
            Dist=Normal,
            # Score=MLE(),
            Base=params["Base"],
            n_estimators=int(params["n_estimators"]),
            learning_rate=params["learning_rate"],
            verbose=False,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return {"loss": np.mean((y_val - preds) ** 2), "status": STATUS_OK}

    b1 = DecisionTreeRegressor(criterion="friedman_mse", max_depth=2)
    b2 = DecisionTreeRegressor(criterion="friedman_mse", max_depth=3)
    b3 = DecisionTreeRegressor(criterion="friedman_mse", max_depth=4)
    space = {
        "Base": hp.choice(
            "Base",
            [
                b1,
                b2,
                b3,
            ],
        ),
        "n_estimators": hp.quniform("n_estimators", 50, 500, 50),
        "learning_rate": hp.uniform("learning_rate", 0.01, 0.1),
    }
    trials = Trials()
    best = fmin(
        fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials
    )
    best_params = space_eval(space, best)
    best_params["n_estimators"] = int(best_params["n_estimators"])
    return best_params


def run():
    mlflow.set_experiment("Daily Timestep")

    y_vals = {}
    ngboost_y_tests = {}

    for target in targets:
        with mlflow.start_run(run_name=f"{target.split()[0]}_ngboost"):
            df = pd.read_csv(
                os.path.join(data_path, f"{target}.csv"),
                parse_dates=["Date"],
            )
            X = df.drop(columns=["Date"] + [target], axis=1)
            y = df[["Date", target]]

            X = X.drop(
                columns=[
                    col
                    for col in X.columns
                    if col.startswith("Calibrated_SWAT") and col not in features[target]
                ]
            )

            # Split the data
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.3, random_state=42, shuffle=False
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42, shuffle=False
            )
            y_vals[target] = y_val
            ngboost_y_tests[target] = y_test

            if target != "Phosphate DlyLd(kg)":
                X = select_features(X_train, y_train[target], target)
                X_train = X_train[X.columns]
                X_val = X_val[X.columns]
                X_test = X_test[X.columns]

            feature_names = list(X.columns)
            mlflow.log_dict(
                {f"input_features": feature_names},
                f"{target.split()[0]}_ngboost_input_features.json",
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)

            mlflow.log_param("target", target)
            best_params = hyperopt_ngboost(
                X_train_scaled, y_train[target], X_val_scaled, y_val[target]
            )
            mlflow.log_params(best_params)

            model = NGBRegressor(Dist=Normal, **best_params)
            model.fit(X_train_scaled, y_train[target])

            preds_test = model.predict(X_test_scaled)
            r2 = r_squared(y_test[target], preds_test)
            nse = nse_score(y_test[target], preds_test)
            pbias = pbias_score(y_test[target], preds_test)
            mae = mean_absolute_error(y_test[target], preds_test)
            mlflow.log_metrics({"R2": r2, "NSE": nse, "PBIAS": pbias, "MAE": mae})

            # Monthly aggregation
            df_month = pd.DataFrame(
                {
                    "Date": y_test["Date"],
                    "Observed": y_test[target],
                    "Predicted": preds_test,
                }
            )
            df_month["Date"] = pd.to_datetime(df_month["Date"])

            # Extract year-month for grouping
            df_month["month"] = df_month["Date"].dt.to_period("M")

            # Aggregate daily loads into monthly totals
            monthly_df = (
                df_month.groupby("month")
                .agg({"Observed": "sum", "Predicted": "sum"})
                .reset_index()
            )

            monthly_r2 = r_squared(monthly_df["Observed"], monthly_df["Predicted"])
            monthly_mae = mean_absolute_error(
                monthly_df["Observed"], monthly_df["Predicted"]
            )
            monthly_nse = nse_score(monthly_df["Observed"], monthly_df["Predicted"])
            monthly_pbias = pbias_score(monthly_df["Observed"], monthly_df["Predicted"])
            monthly_metrics = {}
            monthly_metrics[target] = {
                "R2": monthly_r2,
                "MAE": monthly_mae,
                "NSE": monthly_nse,
                "PBIAS": monthly_pbias,
            }

            mlflow.log_metrics(
                {
                    "Monthly R2": monthly_r2,
                    "Monthly MAE": monthly_mae,
                    "Monthly NSE": monthly_nse,
                    "Monthly PBIAS": monthly_pbias,
                }
            )

            print(f"Monthly metrics for {target}: {monthly_metrics[target]}")


if __name__ == "__main__":
    run()
