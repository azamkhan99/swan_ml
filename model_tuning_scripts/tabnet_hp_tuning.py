# train_tabnet.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_percentage_error,
    r2_score,
    median_absolute_error,
    mean_absolute_error,
)
from pytorch_tabnet.tab_model import TabNetRegressor
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
import mlflow
import torch

from utils import (
    create_lagged_and_cumulative_features,
    nse_score,
    pbias_score,
    r_squared,
    select_features,
    select_top_features_tabnet,
)


import torch.nn.functional as F


def non_negative_mse_loss(y_pred, y_true):
    mse = F.mse_loss(y_pred, y_true)
    penalty = torch.sum(torch.relu(-y_pred))  # Penalize negative predictions
    return mse + 10.0 * penalty  # adjust 10.0 as a hyperparameter


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


# def hyperopt_tabnet(X_train, y_train, X_val, y_val):
#     y_train = y_train.values.reshape(-1, 1)
#     y_val = y_val.values.reshape(-1, 1)

#     def objective(params):
#         model = TabNetRegressor(
#             n_d=int(params["n_d"]),
#             n_a=int(params["n_a"]),
#             n_steps=int(params["n_steps"]),
#             gamma=params["gamma"],
#             lambda_sparse=params["lambda_sparse"],
#             optimizer_fn=torch.optim.Adam,
#             optimizer_params={"lr": params["lr"]},
#             seed=42,
#         )
#         model.fit(
#             X_train=X_train,
#             y_train=y_train,
#             eval_set=[(X_val, y_val)],
#             eval_metric=["rmse"],
#             max_epochs=100,
#             patience=10,
#             batch_size=32,
#         )
#         preds = model.predict(X_val)
#         return {"loss": np.mean((y_val - preds) ** 2), "status": STATUS_OK}


#     space = {
#         "n_d": hp.quniform("n_d", 8, 64, 8),
#         "n_a": hp.quniform("n_a", 8, 64, 8),
#         "n_steps": hp.quniform("n_steps", 3, 10, 1),
#         "gamma": hp.uniform("gamma", 1.0, 2.0),
#         "lambda_sparse": hp.uniform("lambda_sparse", 1e-6, 1e-3),
#         "lr": hp.loguniform("lr", np.log(0.001), np.log(0.1)),
#     }
#     trials = Trials()
#     best = fmin(
#         fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials
#     )
#     return {
#         "n_d": int(best["n_d"]),
#         "n_a": int(best["n_a"]),
#         "n_steps": int(best["n_steps"]),
#         "gamma": best["gamma"],
#         "lambda_sparse": best["lambda_sparse"],
#         "lr": best["lr"],
#     }
def hyperopt_tabnet(X_train, y_train, X_val, y_val):
    y_train = y_train.values.reshape(-1, 1)
    y_val = y_val.values.reshape(-1, 1)

    def objective(params):
        model = TabNetRegressor(
            n_d=int(params["n_d"]),
            n_a=int(params["n_a"]),
            n_steps=int(params["n_steps"]),
            gamma=params["gamma"],
            lambda_sparse=params["lambda_sparse"],
            optimizer_fn=torch.optim.Adam,
            optimizer_params={"lr": params["lr"]},
            scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
            scheduler_params={
                "mode": "min",
                "patience": 5,
                "factor": 0.5,
                "min_lr": 1e-5,
            },
            seed=42,
            verbose=0,
        )

        model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=["rmse"],
            max_epochs=100,
            patience=10,
            batch_size=32,
            drop_last=False,
            loss_fn=non_negative_mse_loss,  # Use custom loss function
        )

        preds = model.predict(X_val)
        loss = mean_squared_error(y_val, preds)
        return {"loss": loss, "status": STATUS_OK}

    space = {
        "n_d": hp.quniform("n_d", 8, 64, 8),
        "n_a": hp.quniform("n_a", 8, 64, 8),
        "n_steps": hp.quniform("n_steps", 3, 10, 1),
        "gamma": hp.uniform("gamma", 1.0, 2.0),
        "lambda_sparse": hp.uniform("lambda_sparse", 1e-6, 1e-3),
        "lr": hp.loguniform("lr", np.log(0.001), np.log(0.1)),
    }

    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=25,
        trials=trials,
        rstate=np.random.default_rng(42),
    )

    return {
        "n_d": int(best["n_d"]),
        "n_a": int(best["n_a"]),
        "n_steps": int(best["n_steps"]),
        "gamma": best["gamma"],
        "lambda_sparse": best["lambda_sparse"],
        "lr": best["lr"],
    }


def run():
    # mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Daily Timestep")

    y_vals = {}
    tabnet_y_tests = {}

    for target in targets:
        with mlflow.start_run(run_name=f"{target.split()[0]}_tabnet_no_doy"):
            df = pd.read_csv(
                os.path.join(data_path, f"{target}.csv"),
                parse_dates=["Date"],
            )
            df["quarter_cos"] = np.cos(2 * np.pi * (df["Date"].dt.quarter - 1) / 4)
            df["quarter_sin"] = np.sin(2 * np.pi * (df["Date"].dt.quarter - 1) / 4)
            X = df.drop(columns=["Date"] + [target], axis=1)
            X.drop(columns=["DOY_cos", "DOY_sin"], inplace=True)
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
            tabnet_y_tests[target] = y_test

            # if target != "Phosphate DlyLd(kg)":
            X = select_top_features_tabnet(X_train, y_train[target], target)

            X_train = X_train[X.columns]
            X_val = X_val[X.columns]
            X_test = X_test[X.columns]

            feature_names = list(X.columns)
            mlflow.log_dict(
                {f"input_features": feature_names},
                f"{target.split()[0]}_tabnet_input_features.json",
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)

            mlflow.sklearn.log_model(scaler, f"{target}_scaler")
            dataset_path = f"mlflow_datasets/{target}_X_train_scaled.csv"
            X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)
            X_train_scaled_df.to_csv(dataset_path, index=False)
            mlflow.log_artifact(dataset_path, artifact_path="datasets")
            dataset = mlflow.data.from_pandas(
                X_train_scaled_df,
                source=dataset_path,
                name=f"{target} Scaled Training Data",
                # targets=f"{target}",
            )

            mlflow.log_param("target", target)
            best_params = hyperopt_tabnet(
                X_train_scaled, y_train[target], X_val_scaled, y_val[target]
            )
            mlflow.log_params(best_params)

            model = TabNetRegressor(
                n_d=best_params["n_d"],
                n_a=best_params["n_a"],
                n_steps=best_params["n_steps"],
                gamma=best_params["gamma"],
                lambda_sparse=best_params["lambda_sparse"],
                optimizer_fn=torch.optim.Adam,
                optimizer_params={"lr": best_params["lr"]},
                seed=42,
                # scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                # scheduler_params={
                #     "mode": "min",
                #     "patience": 5,
                #     "factor": 0.5,
                #     "min_lr": 1e-5,
                # },
            )

            X_train_val = np.concatenate((X_train_scaled, X_val_scaled), axis=0)
            y_train_val = np.concatenate(
                (
                    y_train[target].values.reshape(-1, 1),
                    y_val[target].values.reshape(-1, 1),
                ),
                axis=0,
            )

            model.fit(
                X_train_val,
                y_train_val,
                max_epochs=200,
                patience=15,
                loss_fn=non_negative_mse_loss,  # Use custom loss function
            )

            preds_test = model.predict(X_test_scaled).flatten()
            # print number of negative predictions
            num_neg_preds = np.sum(preds_test < 0)

            print(f"Number of negative predictions: {num_neg_preds}")
            r2 = r_squared(y_test[target], preds_test)
            nse = nse_score(y_test[target], preds_test)
            pbias = pbias_score(y_test[target], preds_test)
            mae = mean_absolute_error(y_test[target], preds_test)
            mape = mean_absolute_percentage_error(y_test[target], preds_test)
            mlflow.log_metrics(
                {"R2": r2, "NSE": nse, "PBIAS": pbias, "MAE": mae, "MAPE": mape}
            )

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
            monthly_mape = mean_absolute_percentage_error(
                monthly_df["Observed"], monthly_df["Predicted"]
            )
            monthly_nse = nse_score(monthly_df["Observed"], monthly_df["Predicted"])
            monthly_pbias = pbias_score(monthly_df["Observed"], monthly_df["Predicted"])
            monthly_metrics = {}
            monthly_metrics[target] = {
                "R2": monthly_r2,
                "NSE": monthly_nse,
                "PBIAS": monthly_pbias,
                "MAE": monthly_mae,
                "MAPE": monthly_mape,
            }

            mlflow.log_metrics(
                {
                    "Monthly R2": monthly_r2,
                    "Monthly NSE": monthly_nse,
                    "Monthly PBIAS": monthly_pbias,
                    "Monthly MAE": monthly_mae,
                    "Monthly MAPE": monthly_mape,
                }
            )

            X_train_val_test = np.concatenate(
                (X_train_scaled, X_val_scaled, X_test_scaled), axis=0
            )
            y_train_val_test = np.concatenate(
                (
                    y_train[target].values.reshape(-1, 1),
                    y_val[target].values.reshape(-1, 1),
                    y_test[target].values.reshape(-1, 1),
                ),
                axis=0,
            )
            full_model = TabNetRegressor(
                n_d=best_params["n_d"],
                n_a=best_params["n_a"],
                n_steps=best_params["n_steps"],
                gamma=best_params["gamma"],
                lambda_sparse=best_params["lambda_sparse"],
                optimizer_fn=torch.optim.Adam,
                optimizer_params={"lr": best_params["lr"]},
                seed=42,
                # scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                # scheduler_params={
                #     "mode": "min",
                #     "patience": 5,
                #     "factor": 0.5,
                #     "min_lr": 1e-5,
                # },
            )
            full_model.fit(
                X_train_val_test,
                y_train_val_test,
                max_epochs=200,
                patience=15,
                loss_fn=non_negative_mse_loss,  # Use custom loss function
            )

            full_model.save_model(f"models/tabnet_no_doy_{target.replace(' ', '_')}")
            print(
                f"Model for {target} saved as models/tabnet_{target.replace(' ', '_')}.zip"
            )
            print(f"Monthly metrics for {target}: {monthly_metrics[target]}")


if __name__ == "__main__":
    run()
