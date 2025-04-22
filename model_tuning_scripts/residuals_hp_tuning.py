import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from pytorch_tabnet.tab_model import TabNetRegressor
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
import mlflow
import torch
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from early_stopping_pytorch import EarlyStopping
from torch.utils.data import DataLoader, TensorDataset

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


# Function to determine the available device
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # elif hasattr(torch, "mps") and torch.backends.mps.is_available():
    #     return torch.device("mps")
    else:
        return torch.device("cpu")


class ImprovedNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.3):
        super(ImprovedNN, self).__init__()
        self.batch_norm_input = nn.BatchNorm1d(input_size)

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_size, output_size)

    def forward(self, x):
        x = self.batch_norm_input(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x


def hyperopt_residual_learning(X_train, y_train, X_val, y_val, device):
    """
    Performs hyperparameter tuning for a feedforward neural network using Hyperopt.

    Args:
        X_train (torch.Tensor): Training features.
        y_train (torch.Tensor): Training target.
        X_val (torch.Tensor): Validation features.
        y_val (torch.Tensor): Validation target.
        device (torch.device): Device to run the model on.

    Returns:
        dict: The best hyperparameters found by Hyperopt.
    """

    def objective(params):
        """
        Defines the objective function to minimize (validation loss).

        Args:
            params (dict): Hyperparameters to evaluate.

        Returns:
            dict: Loss, status, and model.
        """
        # Select model architecture based on number of layers
        hidden_sizes = []
        if params["n_layers"] == 1:
            hidden_sizes = [params["hid_nodes"]]
        elif params["n_layers"] == 2:
            hidden_sizes = [params["hid_nodes"], params["hid_nodes"]]
        else:
            raise ValueError("n_layers must be 1 or 2")

        model = ImprovedNN(
            X_train.shape[1],
            hidden_sizes,
            1,
            dropout_rate=0.3,
        ).to(device)

        criterion = nn.SmoothL1Loss()
        optimizer = optim.Adam(
            model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )

        # Create DataLoader for efficient batching
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, batch_size=params["batch_size"], shuffle=True
        )
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

        num_epochs = 100  # Reduced for hyperparameter search
        best_val_loss = float("inf")
        patience = 10  # Reduced patience for hyperopt
        counter = 0

        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0.0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    val_batch_loss = criterion(outputs, targets).item()
                    val_loss += val_batch_loss

            val_loss /= len(val_loader)
            train_loss /= len(train_loader)

            # Learning rate scheduler step
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    break

        return {"loss": best_val_loss, "status": STATUS_OK, "model": model}

    # Define the search space
    space = {
        "n_layers": hp.choice("n_layers", [1, 2]),  # 1 or 2 hidden layers
        "hid_nodes": hp.choice("hid_nodes", [32, 64, 128]),
        "lr": hp.loguniform("lr", -5, -2),  # Logarithmic scale for learning rate
        "batch_size": hp.choice("batch_size", [16, 32, 64]),
        "weight_decay": hp.loguniform("weight_decay", -6, -2),
    }

    # Run Hyperopt
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=50,  # Reduced number of evaluations for faster tuning
        trials=trials,
    )

    # Extract the best hyperparameters
    hid_node_choices = [32, 64, 128]
    batch_size_choices = [16, 32, 64]

    best_params = {
        "n_layers": best["n_layers"] + 1,  # Convert choice index to actual value
        "hid_nodes": hid_node_choices[best["hid_nodes"]],
        "lr": best["lr"],
        "batch_size": batch_size_choices[best["batch_size"]],
        "weight_decay": best["weight_decay"],
        "hidden_sizes": [hid_node_choices[best["hid_nodes"]]]
        * (best["n_layers"] + 1),  # Create hidden_sizes list
    }
    return best_params


def run(log_transform=False):
    mlflow.set_experiment("Daily Timestep")

    targets = [
        "Phosphate DlyLd(kg)",
        "Nitrate DlyLd(kg)",
        "Sediments DlyLd(kg*1000)",
    ]

    # Detect and use the appropriate device
    device = get_device()
    print(f"Using device: {device}")

    y_vals = {}
    y_tests = {}
    y_transformers = {}
    y_actuals = {}
    model_test_preds = {}

    pbar = tqdm(total=len(targets), desc="Processing targets")

    for target in targets:
        pbar.set_description(f"Processing {target}")
        if log_transform:
            experiment_run_name = f"{target.split()[0]}_residual_log"
        else:
            experiment_run_name = f"{target.split()[0]}_residual"
        with mlflow.start_run(run_name=experiment_run_name):
            torch.manual_seed(10)
            df = pd.read_csv(
                os.path.join(data_path, f"{target}.csv"),
                parse_dates=["Date"],
            )
            X = df.drop(columns=["Date"] + [target], axis=1)
            y = df[target] - df[f"Calibrated_SWAT_{target}"]

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

            feature_names = list(X.columns)
            mlflow.log_dict(
                {f"input_features": feature_names},
                f"{target.split()[0]}_residual_input_features.json",
            )

            scaler = QuantileTransformer(output_distribution="normal")
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)

            mlflow.log_param("target", target)

            if log_transform:
                # Add a small constant to handle negative values
                min_val = min(y_train.min(), y_val.min())
                offset = abs(min_val) + 1 if min_val < 0 else 0

                y_train_log = np.log1p(y_train + offset)
                y_val_log = np.log1p(y_val + offset)

                # Scale the log-transformed values
                residual_transformer = StandardScaler()
                y_train_transformed = residual_transformer.fit_transform(
                    y_train_log.values.reshape(-1, 1)
                ).flatten()
                y_val_transformed = residual_transformer.transform(
                    y_val_log.values.reshape(-1, 1)
                ).flatten()

                # Store transformer and offset for later inverse transform
                y_transformers[target] = {
                    "scaler": residual_transformer,
                    "offset": offset,
                    "log_transform": True,
                }
            else:
                # Standard scaling for normally distributed residuals
                residual_transformer = StandardScaler()
                y_train_transformed = residual_transformer.fit_transform(
                    y_train.values.reshape(-1, 1)
                ).flatten()
                y_val_transformed = residual_transformer.transform(
                    y_val.values.reshape(-1, 1)
                ).flatten()

                # Store transformer for later inverse transform
                y_transformers[target] = {
                    "scaler": residual_transformer,
                    "offset": 0,
                    "log_transform": False,
                }

            # Convert to PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
            y_train_tensor = torch.FloatTensor(y_train_transformed.reshape(-1, 1)).to(
                device
            )
            X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
            y_val_tensor = torch.FloatTensor(y_val_transformed.reshape(-1, 1)).to(
                device
            )
            X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)

            # Create dataloaders with smaller batch size
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=16)

            best_params = hyperopt_residual_learning(
                X_train_tensor,
                y_train_tensor,
                X_val_tensor,
                y_val_tensor,
                device,
            )

            mlflow.log_params(best_params)

            input_size = X_train.shape[1]
            output_size = 1

            model = ImprovedNN(
                input_size, best_params["hidden_sizes"], output_size, dropout_rate=0.3
            ).to(device)

            criterion = nn.SmoothL1Loss()
            optimizer = optim.Adam(
                model.parameters(),
                lr=best_params["lr"],
                weight_decay=best_params["weight_decay"],
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=10
            )

            # Train the model
            num_epochs = 200
            best_val_loss = float("inf")
            patience = 20
            counter = 0

            for epoch in range(num_epochs):
                # Training
                model.train()
                train_loss = 0.0
                for inputs, targets in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

            # Learning rate scheduling
            avg_val_loss = val_loss / len(val_loader)
            scheduler.step(avg_val_loss)

            # Update progress bar status
            avg_train_loss = train_loss / len(train_loader)
            pbar.set_postfix(
                {
                    "target": target,
                    "epoch": f"{epoch + 1}/{num_epochs}",
                    "train_loss": f"{avg_train_loss:.4f}",
                    "val_loss": f"{avg_val_loss:.4f}",
                }
            )

            # Test set evaluation
            model.eval()
            with torch.no_grad():
                y_pred_test_transformed = model(X_test_tensor).cpu().numpy()

            # Inverse transform the predictions to get back to original scale
            if y_transformers[target]["log_transform"]:
                # Inverse scaling
                y_pred_test_scaled = (
                    y_transformers[target]["scaler"]
                    .inverse_transform(y_pred_test_transformed)
                    .flatten()
                )
                # Inverse log transform
                y_pred_test = (
                    np.expm1(y_pred_test_scaled) - y_transformers[target]["offset"]
                )
            else:
                # Just inverse scaling
                y_pred_test = (
                    y_transformers[target]["scaler"]
                    .inverse_transform(y_pred_test_transformed)
                    .flatten()
                )

            y_tests[target] = y_test

            # Compute final corrected predictions by adding back the SWAT predictions
            y_pred_corrected = (
                df[f"Calibrated_SWAT_{target}"].loc[y_test.index] + y_pred_test
            )
            model_test_preds[target] = y_pred_corrected

            # Get actual observations
            y_actual = df[["Date", target]].loc[y_test.index]
            y_actuals[target] = y_actual

            r2 = r_squared(y_actual[target], y_pred_corrected)
            nse = nse_score(y_actual[target], y_pred_corrected)
            pbias = pbias_score(y_actual[target], y_pred_corrected)
            mae = mean_absolute_error(y_actual[target], y_pred_corrected)
            mlflow.log_metrics({"R2": r2, "NSE": nse, "PBIAS": pbias, "MAE": mae})

            pbar.update(1)

            # Monthly aggregation
            df_month = pd.DataFrame(
                {
                    "Date": y_actuals[target]["Date"],
                    "Observed": y_actuals[target][target],
                    "Predicted": model_test_preds[target],
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

            # Save model to CPU before saving to disk
            model.to("cpu")
            torch.save(
                model.state_dict(), f"models/residual_{target.replace(' ', '_')}.pt"
            )
            print(
                f"Model for {target} saved as models/residual_{target.replace(' ', '_')}.pt"
            )
            print(f"Monthly metrics for {target}: {monthly_metrics[target]}")
            pbar.close()


if __name__ == "__main__":
    # Set the log transformation flag through command line argument
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_transform",
        action="store_true",
        help="Apply log transformation to the target variable",
    )
    args = parser.parse_args()
    log_transform = args.log_transform
    print(f"Log transformation applied: {log_transform}")
    # Run the main function
    run(log_transform=log_transform)
