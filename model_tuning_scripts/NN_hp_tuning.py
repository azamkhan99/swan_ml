import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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


class learnKappa_layers1(nn.Module):
    def __init__(self, In_nodes, Hid, Out_nodes):
        super(learnKappa_layers1, self).__init__()
        self.linear1 = nn.Linear(In_nodes, Hid)  # Input to hidden layer
        self.linear2 = nn.Linear(Hid, Out_nodes)  # Hidden to output layer
        self.dropout = nn.Dropout(0.20)  # Dropout to reduce overfitting

    def forward(self, x):
        x2 = self.linear1(x)
        h1 = torch.relu(x2)  # ReLU activation
        h1 = self.dropout(h1)
        y_pred = self.linear2(h1)  # Output predictions
        return y_pred


class learnKappa_layers2(nn.Module):
    def __init__(self, In_nodes, Hid, Out_nodes):
        super(learnKappa_layers2, self).__init__()
        self.linear1 = nn.Linear(In_nodes, Hid)
        self.linear2 = nn.Linear(Hid, Hid)
        self.linear3 = nn.Linear(Hid, Out_nodes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x2 = self.linear1(x)
        h1 = torch.relu(x2)
        h1 = self.dropout(h1)
        h2 = self.linear2(h1)
        h3 = torch.relu(h2)
        h3 = self.dropout(h3)
        y_pred = self.linear3(h3)
        return y_pred


def hyperopt_ffnn(X_train, y_train, X_val, y_val, device):
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
        # Select model architecture
        if params["n_layers"] == 1:
            model = learnKappa_layers1(
                X_train.shape[1],
                params["hid_nodes"],
                1,
            ).to(device)
        elif params["n_layers"] == 2:
            model = learnKappa_layers2(
                X_train.shape[1],
                params["hid_nodes"],
                1,
            ).to(device)
        else:
            raise ValueError("n_layers must be 1 or 2")

        criterion = nn.SmoothL1Loss()
        optimizer = optim.Adam(
            model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
        )
        epochs = 200  # Reduced number of epochs for faster tuning

        # Create DataLoader for efficient batching
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, batch_size=params["batch_size"], shuffle=True
        )
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        for epoch in range(epochs):
            model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                y_pred = model(batch_x).reshape(-1)
                loss = criterion(y_pred, batch_y)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    y_pred = model(batch_x).reshape(-1)
                    val_loss += criterion(y_pred, batch_y).item()
                val_loss /= len(val_loader)

        return {"loss": val_loss, "status": STATUS_OK, "model": model}

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
    best_params = {
        "n_layers": best["n_layers"] + 1,  # Convert choice index to actual value
        "hid_nodes": [16, 32, 64, 128][best["hid_nodes"]],
        "lr": best["lr"],
        "batch_size": [16, 32, 64][best["batch_size"]],
        "weight_decay": best["weight_decay"],
    }
    return best_params


def modeltrain_loss(
    in_nodes,
    hid_nodes,
    out_nodes,
    lr,
    batch_size,
    weight_decay,
    epochs,
    X_train,
    y_train,
    X_val,
    y_val,
    model,
    device,
    patience=400,
):
    criterion = nn.SmoothL1Loss()  # Loss function
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )  # Optimizer with weight decay
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # Create DataLoader for efficient batching
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )  # Use batch_size
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Create arrays to store loss values
    train_losses = []
    val_losses = []

    with tqdm(total=epochs) as pbar:
        for epoch in range(epochs):
            # Training mode
            model.train()
            batch_losses = []
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                y_pred = model(batch_x)
                # Reshape y_pred if needed to match y_train
                y_pred = y_pred.reshape(-1)
                loss = criterion(y_pred, batch_y)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())

            # Average batch loss for the epoch
            avg_train_loss = sum(batch_losses) / len(batch_losses)
            train_losses.append(avg_train_loss)

            # Validation mode
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    y_val_pred = model(batch_x)
                    # Reshape y_val_pred if needed to match y_val
                    y_val_pred = y_val_pred.reshape(-1)
                    val_loss += criterion(y_val_pred, batch_y).item()
                val_loss /= len(val_loader)
                val_losses.append(val_loss)

            # Update progress bar
            pbar.update(1)
            pbar.set_description(
                f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

            # Early stopping
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    return model, (train_losses, val_losses)


def run(log_transform=False):
    mlflow.set_experiment("Daily Timestep_bigger_dataset")

    # Detect and use the appropriate device
    device = get_device()
    print(f"Using device: {device}")

    y_vals = {}
    ffnn_y_tests = {}

    for target in targets:
        if log_transform:
            experiment_run_name = f"{target.split()[0]}_ffnn_log"
        else:
            experiment_run_name = f"{target.split()[0]}_ffnn"
        with mlflow.start_run(run_name=experiment_run_name):
            torch.manual_seed(10)
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
                X, y, test_size=0.40, random_state=42, shuffle=False
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.75, random_state=42, shuffle=False
            )
            y_vals[target] = y_val
            ffnn_y_tests[target] = y_test

            feature_names = list(X.columns)
            mlflow.log_dict(
                {f"input_features": feature_names},
                f"{target.split()[0]}_ffnn_input_features.json",
            )

            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)

            mlflow.log_param("target", target)

            if log_transform:
                # Add a small constant to handle negative values
                min_val = min(y_train[target].min(), y_val[target].min())
                offset = abs(min_val) + 1 if min_val < 0 else 0
                y_train_log = np.log1p(y_train[target] + offset).values
                y_val_log = np.log1p(y_val[target] + offset).values
                y_test_log = np.log1p(
                    y_test[target] + offset
                ).values  # Also transform test

                y_train_tensor = torch.tensor(y_train_log, dtype=torch.float32).to(
                    device
                )
                y_val_tensor = torch.tensor(y_val_log, dtype=torch.float32).to(device)
                y_test_tensor = torch.tensor(y_test_log, dtype=torch.float32).to(
                    device
                )  # Use transformed test

                # Store the offset for inverse transformation
                y_transformers = {target: {"offset": offset, "log_transform": True}}
            else:
                y_train_tensor = torch.tensor(
                    y_train[target].values, dtype=torch.float32
                ).to(device)
                y_val_tensor = torch.tensor(
                    y_val[target].values, dtype=torch.float32
                ).to(device)
                y_test_tensor = torch.tensor(
                    y_test[target].values, dtype=torch.float32
                ).to(device)
                y_transformers = {target: {"offset": 0, "log_transform": False}}

            X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(
                device
            )
            X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

            best_params = hyperopt_ffnn(
                X_train_tensor,
                y_train_tensor,
                X_val_tensor,
                y_val_tensor,
                device,
            )

            mlflow.log_params(best_params)

            epochs = 5000

            if best_params["n_layers"] == 1:
                model = learnKappa_layers1(
                    X_train_scaled.shape[1],
                    best_params["hid_nodes"],
                    1,
                ).to(device)
            else:
                model = learnKappa_layers2(
                    X_train_scaled.shape[1],
                    best_params["hid_nodes"],
                    1,
                ).to(device)
            model, loss_arrays = modeltrain_loss(
                in_nodes=X_train_scaled.shape[1],
                hid_nodes=best_params["hid_nodes"],
                out_nodes=1,
                lr=best_params["lr"],
                batch_size=best_params["batch_size"],
                weight_decay=best_params["weight_decay"],
                epochs=epochs,
                X_train=X_train_tensor,
                y_train=y_train_tensor,
                X_val=X_val_tensor,
                y_val=y_val_tensor,
                model=model,
                device=device,
                patience=500,
            )
            model.eval()
            with torch.no_grad():
                test_predictions = model(X_test_tensor).reshape(-1).cpu().numpy()

            # Inverse transform predictions
            if y_transformers[target]["log_transform"]:
                preds_test = (
                    np.expm1(test_predictions) - y_transformers[target]["offset"]
                )
            else:
                preds_test = test_predictions

            r2 = r_squared(y_test[target], preds_test)
            nse = nse_score(y_test[target], preds_test)
            pbias = pbias_score(y_test[target], preds_test)
            mae = mean_absolute_error(y_test[target], preds_test)
            mlflow.log_metrics({"R2": r2, "NSE": nse, "PBIAS": pbias, "MAE": mae})

            # Monthly aggregation
            df_month = pd.DataFrame(
                {
                    "Date": ffnn_y_tests[target]["Date"],
                    "Observed": ffnn_y_tests[target][target],
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

            # Save model to CPU before saving to disk
            model.to("cpu")
            torch.save(model.state_dict(), f"models/ffnn_{target.replace(' ', '_')}.pt")
            print(
                f"Model for {target} saved as models/ffnn_{target.replace(' ', '_')}.pt"
            )
            print(f"Monthly metrics for {target}: {monthly_metrics[target]}")


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
