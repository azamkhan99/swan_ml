import mlflow.artifacts
import xgboost as xgb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pprint import pprint

from sklearn.preprocessing import StandardScaler
from scipy.stats import randint as sp_randint

import mlflow
import hyperopt
import json
import shap

from xgboost import XGBRegressor
import logging
import os

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

from utils import (
    create_lagged_and_cumulative_features,
    nse_score,
    pbias_score,
    r_squared,
    # select_features,
)


def date_feature_transformations(dataset):
    dataset["Date"] = pd.to_datetime(dataset["Date"])
    dataset["month"] = dataset["Date"].dt.month
    dataset["DOY"] = dataset["Date"].dt.dayofyear
    dataset["DOY_cos"] = np.cos(2 * np.pi * dataset["DOY"] / 365)
    dataset["DOY_sin"] = np.sin(2 * np.pi * dataset["DOY"] / 365)
    dataset["month_cos"] = np.cos(2 * np.pi * dataset["month"] / 12)
    dataset["month_sin"] = np.sin(2 * np.pi * dataset["month"] / 12)
    dataset["Temp_range"] = dataset["MaxT"] - dataset["MinT"]
    dataset.drop(["DOY"], axis=1, inplace=True)
    dataset.drop(["month"], axis=1, inplace=True)
    return dataset


def select_features(target, run_id_model_mapping):
    run_id = run_id_model_mapping[target]
    artifact_path = f"{target.split(' ')[0]}_residual_input_features.json"
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact_path
    )
    with open(local_path, "r") as f:
        loaded_dict = json.load(f)
    return loaded_dict["input_features"]


def load_scaler(target, run_id_model_mapping, scaler_type=None):
    if scaler_type is not None:
        run_id = run_id_model_mapping[target]
        logged_model = f"runs:/{run_id}/{target}_residual_transformer"
        return mlflow.sklearn.load_model(logged_model)

    run_id = run_id_model_mapping[target]
    logged_model = f"runs:/{run_id}/{target}_scaler"
    return mlflow.sklearn.load_model(logged_model)


def load_offset_param(target, run_id_model_mapping):
    run_id = run_id_model_mapping[target]
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    params = run.data.params
    # Extract the offset parameter
    offset_param = params.get("logOffset")
    return float(offset_param)


def main():
    # Define run ID mapping
    mlflow.set_tracking_uri("file:///Users/azamkhan/columbia/climate/swan_ml/mlruns")
    run_id_model_mapping = {
        "Sediments DlyLd(kg*1000)": "3c6199a4e6894176ba7b1f4c19a0dc2e",
        "Nitrate DlyLd(kg)": "8bca5e53495649b887a3c0ac4ff7c1f9",
        "Phosphate DlyLd(kg)": "936470d2fef44bfdbd4922386b9e1c45",
    }

    climate_models = [
        "GFDL-ESM",
        "IPSL-CM6A-LR",
        "MPI-ESM1-2-HR",
        "MRI-ESM2-0",
        # "UKESM1-0-LL",
    ]
    targets = [
        "Sediments DlyLd(kg*1000)",
        "Nitrate DlyLd(kg)",
        "Phosphate DlyLd(kg)",
    ]

    swat_column_mapping = {
        "FLOW_OUTcms": "Calibrated_SWAT_Streamflow",
        "SED_OUTtons": "Calibrated_SWAT_Sediments DlyLd(kg*1000)",
        "NO3_OUTkg": "Calibrated_SWAT_Nitrate DlyLd(kg)",
        "MINP_OUTkg": "Calibrated_SWAT_Phosphate DlyLd(kg)",
        "ETmm": "Evapotranspiration",
        "SNOMELTmm": "sim_snowmelt",
    }

    climate_column_mapping = {
        "TMIN": "MinT",
        "TMAX": "MaxT",
        "RAIN": "Precipitation",
    }

    emission_scenarios = ["HighEmission", "LowEmission"]
    # emission_scenarios = ["HighEmission"]
    # Input file paths
    # climate_file = input("Enter the path to the climate data Excel file: ")
    # swat_file = input("Enter the path to the SWAT simulation Excel file: ")

    for climate_model in climate_models:
        for emission_scenario in emission_scenarios:
            climate_file = f"/Users/azamkhan/columbia/climate/Extractions/FutureClimateData/{climate_model}/{emission_scenario}/future_climate_variables.csv"
            swat_file_quals = f"/Users/azamkhan/columbia/climate/Extractions/Daily_CC_Runs/{climate_model}/{emission_scenario}/SWAT_OUTPUT/rch6_swat_output.csv"
            swat_file_et = f"/Users/azamkhan/columbia/climate/Extractions/Daily_CC_Runs/{climate_model}/{emission_scenario}/ET/sub6_et.csv"
            swat_file_snowmelt = f"/Users/azamkhan/columbia/climate/Extractions/Daily_CC_Runs/{climate_model}/{emission_scenario}/SM/snowmelt.csv"
            swat_file = "s"

            # Load datasets
            logging.info("Loading Future climate data...")
            climate_df = pd.read_csv(climate_file)
            climate_df["Date"] = pd.to_datetime(climate_df["Date"], utc=True)
            climate_df.rename(columns=climate_column_mapping, inplace=True)

            logging.info("Loading SWAT data...")
            simulated_quals = pd.read_csv(swat_file_quals)
            simulated_quals.rename(columns={"Unnamed: 0": "Date"}, inplace=True)
            simulated_quals["Date"] = pd.to_datetime(simulated_quals["Date"], utc=True)
            simulated_et = pd.read_csv(swat_file_et)
            simulated_et.rename(columns={"Unnamed: 0": "Date"}, inplace=True)
            simulated_et["Date"] = pd.to_datetime(simulated_et["Date"], utc=True)
            simulated_snowmelt = pd.read_csv(swat_file_snowmelt)
            simulated_snowmelt.rename(columns={"Unnamed: 0": "Date"}, inplace=True)
            simulated_snowmelt["Date"] = pd.to_datetime(
                simulated_snowmelt["Date"], utc=True
            )

            simulated = pd.merge(simulated_quals, simulated_et, on="Date", how="inner")
            simulated = pd.merge(simulated, simulated_snowmelt, on="Date", how="inner")
            simulated.drop(columns=["ORGP_OUTkg"], inplace=True)
            simulated.rename(columns=swat_column_mapping, inplace=True)

            swat_quals_dfs = {}
            for target in targets:
                swat_quals_dfs[target] = simulated

            logging.info("Merging datasets...")

            # water_quality_datasets = {}
            for key in swat_quals_dfs:
                # swat_quals_dfs[key] = swat_quals_dfs[key].merge(
                #     simulated, on="Date", how="left"
                # )
                swat_quals_dfs[key] = swat_quals_dfs[key].merge(
                    climate_df, on="Date", how="inner"
                )

                logging.info(f"{emission_scenario}: {swat_quals_dfs[key].head(1)}")

                # water_quality_datasets[key] = swat_quals_dfs.pop(key)
            # print(swat_quals_dfs["Sediments DlyLd(kg*1000)"].columns)

            for target in swat_quals_dfs.keys():
                logging.info(f"Processing target: {target}")

                dataset = swat_quals_dfs[target]
                dataset = date_feature_transformations(dataset)
                dataset = create_lagged_and_cumulative_features(
                    dataset,
                    lag_days=[1, 3, 7],
                    cumulation_days=[3, 7],
                )

                X = dataset.drop(columns=["Date"], axis=1)
                # y = dataset[target]

                input_features = select_features(target, run_id_model_mapping)
                X = X[input_features]

                scaler = load_scaler(target, run_id_model_mapping)
                X = scaler.transform(X)
                X_tensor = torch.tensor(X, dtype=torch.float32)

                mlflow_model_uri = f"runs:/{run_id_model_mapping[target]}/{target.split(' ')[0]}_residual_model"

                loaded_model = mlflow.pytorch.load_model(mlflow_model_uri)

                # X_train_path = f"mlruns/806858634930860684/{run_id_model_mapping[target]}/artifacts/dataset/{target}_X_train_scaled.csv"
                # X_train = pd.read_csv(X_train_path)
                # background_data = X_train.sample(100, random_state=42)

                # def predict_fn(x):
                #     return loaded_model.predict(x)

                # explainer = shap.KernelExplainer(predict_fn, background_data)

                # def explain_prediction(new_data):
                #     shap_values = explainer.shap_values(new_data)
                #     return shap_values

                # logging.info("Explaining predictions...")
                # sv = explainer(X)
                loaded_model.eval()
                with torch.no_grad():
                    y_pred_transformed = (
                        loaded_model(X_tensor).reshape(-1).cpu().numpy()
                    )
                    residual_transformer = load_scaler(
                        target, run_id_model_mapping, scaler_type="residual"
                    )
                    y_pred_test_scaled = residual_transformer.inverse_transform(
                        y_pred_transformed.reshape(-1, 1)
                    ).flatten()
                    # inverse log transform
                    logOffset = load_offset_param(target, run_id_model_mapping)
                    y_pred = np.expm1(y_pred_test_scaled) - logOffset

                    y_pred_corrected = (
                        dataset[f"Calibrated_SWAT_{target}"].values + y_pred
                    )
                # y_pred = sv.data

                inference_results_df = pd.DataFrame(
                    {
                        "Date": dataset["Date"],
                        "Predicted": y_pred_corrected,
                    }
                )

                logging.info("Writing results to CSV...")
                # results_df.to_csv("results.csv", index=True)
                output_dir = f"ml_prediction_outputs_residual/{climate_model}/{emission_scenario}"
                # logging.info(
                #     f"Writing results to {output_dir}/{target.split(' ')[0]}_raw_predictions.csv..."
                # )
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                inference_results_df.to_csv(
                    f"{output_dir}/{target.split(' ')[0]}_raw_predictions.csv",
                    index=False,
                )

                # shap_output_dir = f"ml_prediction_outputs/{climate_model}/{emission_scenario}/shap_values"
                # if not os.path.exists(shap_output_dir):
                #     os.makedirs(shap_output_dir)
                # logging.info(
                #     f"Writing SHAP values to {shap_output_dir}/{target.split(' ')[0]}_shap_values.csv..."
                # )
                # shap_values_df = pd.DataFrame(
                #     np.c_[sv.base_values, sv.values], columns=["bv"] + input_features
                # )
                # shap_values_df["Date"] = dataset["Date"]
                # shap_values_df.to_csv(
                #     f"{shap_output_dir}/{target.split(' ')[0]}_shap_values.csv",
                #     index=False,
                # )

                logging.info(f"{target}: {inference_results_df.head(1)}")

            logging.info(
                f"Inference results written to CSV for {climate_model} {emission_scenario}."
            )


if __name__ == "__main__":
    main()
