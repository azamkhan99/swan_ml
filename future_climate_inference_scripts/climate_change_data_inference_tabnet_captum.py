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

from pytorch_tabnet.tab_model import TabNetRegressor
from captum.attr import IntegratedGradients
import torch
import logging
import os

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


def attribute_with_tqdm(ig_explainer, X_tensor, target=0, batch_size=32):
    from tqdm import tqdm

    all_attributions = []
    all_deltas = []
    n = X_tensor.shape[0]

    for i in tqdm(range(0, n, batch_size), desc="Attributing with IG"):
        batch = X_tensor[i : i + batch_size]
        attr, delta = ig_explainer.attribute(
            batch, target=target, return_convergence_delta=True
        )
        all_attributions.append(attr)
        all_deltas.append(delta)

    all_attributions = torch.cat(all_attributions, dim=0)
    all_deltas = torch.cat(all_deltas, dim=0)
    return all_attributions, all_deltas


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def date_feature_transformations(dataset):
    dataset["Date"] = pd.to_datetime(dataset["Date"])
    dataset["month"] = dataset["Date"].dt.month
    dataset["DOY"] = dataset["Date"].dt.dayofyear
    # dataset["DOY_cos"] = np.cos(2 * np.pi * dataset["DOY"] / 365)
    # dataset["DOY_sin"] = np.sin(2 * np.pi * dataset["DOY"] / 365)
    dataset["month_cos"] = np.cos(2 * np.pi * dataset["month"] / 12)
    dataset["month_sin"] = np.sin(2 * np.pi * dataset["month"] / 12)
    dataset["quarter_cos"] = np.cos(2 * np.pi * dataset["Date"].dt.quarter / 4)
    dataset["quarter_sin"] = np.sin(2 * np.pi * dataset["Date"].dt.quarter / 4)
    dataset["Temp_range"] = dataset["MaxT"] - dataset["MinT"]
    dataset.drop(["DOY"], axis=1, inplace=True)
    dataset.drop(["month"], axis=1, inplace=True)
    return dataset


def select_features(target, run_id_model_mapping):
    run_id = run_id_model_mapping[target]
    artifact_path = f"{target.split(' ')[0]}_tabnet_input_features.json"
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact_path
    )
    with open(local_path, "r") as f:
        loaded_dict = json.load(f)
    return loaded_dict["input_features"]


def load_scaler(target, run_id_model_mapping):
    run_id = run_id_model_mapping[target]
    logged_model = f"runs:/{run_id}/{target}_scaler"
    return mlflow.sklearn.load_model(logged_model)


def main():
    # Define run ID mapping
    mlflow.set_tracking_uri("file:///Users/azamkhan/columbia/climate/swan_ml/mlruns")
    run_id_model_mapping = {
        "Sediments DlyLd(kg*1000)": "7dbc2f38bdda4a7b865971cfbc3b0b9e",
        "Nitrate DlyLd(kg)": "f7a1076fd4214612822c27002eaaf017",
        "Phosphate DlyLd(kg)": "ee57487119f54539967d5002b8ac6268",
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
                X_scaled = scaler.transform(X)

                # --- Load your TabNet model (PyTorch native model, not pyfunc wrapper) ---
                # tabnet_model_uri = f"runs:/{run_id_model_mapping[target]}/models/tabnet_{target.replace(' ', '_')}"
                tabnet_model_path = (
                    f"../models/tabnet_no_doy_{target.replace(' ', '_')}.zip"
                )
                # tabnet_model = mlflow.pyfunc.load_model(tabnet_model_uri)

                # # now load model as a pytorch tabnet model
                device = get_device()
                tabnet_model = TabNetRegressor(device_name="mps")
                tabnet_model.load_model(tabnet_model_path)

                # --- TabNet expects torch.FloatTensor ---
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

                network = tabnet_model.network
                network.eval()

                # tabnet_model.network.eval()

                # --- Create Captum Integrated Gradients Explainer ---

                logging.info(
                    f"Explaining predictions for {target} using Integrated Gradients..."
                )

                def clean_forward(x):
                    outputs, _ = network(x)
                    return outputs

                ig = IntegratedGradients(clean_forward)

                # --- Compute attributions ---
                # attributions, delta = ig.attribute(
                #     X_tensor,
                #     target=0,
                #     return_convergence_delta=True,
                # )
                attributions, delta = attribute_with_tqdm(
                    ig, X_tensor, target=0, batch_size=32
                )

                # --- Predictions ---
                with torch.no_grad():
                    outputs, M_loss = network(X_tensor)
                    y_pred = outputs.cpu().numpy().flatten()

                # --- Save predictions ---
                inference_results_df = pd.DataFrame(
                    {
                        "Date": dataset["Date"],
                        "Predicted": y_pred,
                    }
                )

                output_dir = (
                    f"ml_prediction_outputs_no_doy/{climate_model}/{emission_scenario}"
                )
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                inference_results_df.to_csv(
                    f"{output_dir}/{target.split(' ')[0]}_raw_predictions.csv",
                    index=False,
                )

                # --- Save attributions ---
                shap_output_dir = f"ml_prediction_outputs_no_doy/{climate_model}/{emission_scenario}/shap_values"
                if not os.path.exists(shap_output_dir):
                    os.makedirs(shap_output_dir)

                attributions_np = attributions.detach().cpu().numpy()
                attributions_df = pd.DataFrame(attributions_np, columns=input_features)
                attributions_df["Date"] = dataset["Date"]

                logging.info(
                    f"Saving feature attributions to {shap_output_dir}/{target.split(' ')[0]}_attributions.csv"
                )
                attributions_df.to_csv(
                    f"{shap_output_dir}/{target.split(' ')[0]}_attributions.csv",
                    index=False,
                )

                logging.info(f"{target}: {inference_results_df.head(1)}")

                logging.info(
                    f"Inference results and attributions written to CSV for {climate_model} {emission_scenario}."
                )


if __name__ == "__main__":
    main()
