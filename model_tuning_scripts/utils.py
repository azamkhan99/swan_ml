import numpy as np
from scipy.stats import pearsonr
from xgboost import XGBRegressor
import pandas as pd


def nse_score(observed, simulated):
    """
    Calculate the Nash-Sutcliffe Efficiency (NSE) between observed and simulated data.

    Parameters:
    observed (array-like): Array of observed values.
    simulated (array-like): Array of simulated/predicted values.

    Returns:
    float: NSE value.
    """
    observed = np.asarray(observed)
    simulated = np.asarray(simulated)

    if observed.shape != simulated.shape:
        raise ValueError("Observed and simulated arrays must have the same shape.")

    numerator = np.sum((simulated - observed) ** 2)
    denominator = np.var(observed, ddof=1) * len(observed)

    nse = 1 - numerator / denominator
    return nse


def pbias_score(observed, predicted):
    """
    Calculate the percentage difference between sum of observed and sum of predicted values.

    Parameters:
    observed (numpy.ndarray): Array of observed values
    predicted (numpy.ndarray): Array of predicted values

    Returns:
    float: Percentage difference: (sum(observed) - sum(predicted)) * 100 / sum(observed)
    """
    sum_observed = np.sum(observed)
    sum_predicted = np.sum(predicted)

    if sum_observed == 0:
        raise ValueError("Sum of observed values cannot be zero (division by zero)")

    percentage_diff = (sum_observed - sum_predicted) * 100 / sum_observed

    return percentage_diff


def r_squared(observed, simulated):
    return pearsonr(observed, simulated)[0] ** 2


def select_features(X, y, target_column, importance_threshold=0.01):
    # Fit a preliminary model to get feature importances
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=50,
        learning_rate=0.1,
        max_depth=4,
        random_state=42,
    )
    model.fit(X, y)

    # Extract feature importances
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)

    # Filter features based on importance threshold
    important_features = feature_importances[
        feature_importances > importance_threshold
    ].index
    print(f"Selected features for {target_column}:\n", important_features)

    # Return data with selected features only
    return X[important_features]


from pytorch_tabnet.tab_model import TabNetRegressor
import pandas as pd
import numpy as np


def select_top_features_tabnet(X, y, target_column, top_n=25):
    # Convert to numpy arrays
    X_np = X.values
    y_np = y.values.reshape(-1, 1)

    # Train TabNet for feature importance extraction
    model = TabNetRegressor(seed=42)
    model.fit(
        X_np,
        y_np,
        max_epochs=100,
        patience=10,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
    )

    # Get and rank feature importances
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    top_features = feature_importances.sort_values(ascending=False).head(top_n).index

    if "Calibrated_SWAT_Streamflow" not in top_features:
        top_features = top_features.insert(0, "Calibrated_SWAT_Streamflow")
        # delete the last element to maintain the top_n count
        top_features = top_features[:-1]

    print(f"Top {top_n} features for {target_column}:\n", top_features.tolist())

    return X[top_features]


def create_lagged_and_cumulative_features(dataset, lag_days, cumulation_days):
    """
    Creates lagged and cumulative features for specified columns in a Pandas DataFrame.

    Args:
        dataset (pd.DataFrame): The input DataFrame.
        lag_days (list): List of integers representing the lag days.
        cumulation_days (list): List of integers representing the cumulation days.

    Returns:
        pd.DataFrame: The DataFrame with added lagged and cumulative features.
    """

    new_dataset = dataset.copy()

    # Create lagged features
    for lag in lag_days:
        new_dataset[f"Precipitation_lag{lag}"] = new_dataset["Precipitation"].shift(lag)
        new_dataset[f"Evapotranspiration_lag{lag}"] = new_dataset[
            "Evapotranspiration"
        ].shift(lag)
        new_dataset[f"snowmelt_lag{lag}"] = new_dataset["sim_snowmelt"].shift(lag)
        new_dataset[f"MinT_lag{lag}"] = new_dataset["MinT"].shift(lag)
        new_dataset[f"MaxT_lag{lag}"] = new_dataset["MaxT"].shift(lag)

    # Create cumulative features
    for cumulation in cumulation_days:
        new_dataset[f"Precipitation_cum{cumulation}"] = (
            new_dataset["Precipitation"].rolling(cumulation).sum()
        )
        new_dataset[f"Evapotranspiration_cum{cumulation}"] = (
            new_dataset["Evapotranspiration"].rolling(cumulation).sum()
        )
        new_dataset[f"snowmelt_cum{cumulation}"] = (
            new_dataset["sim_snowmelt"].rolling(cumulation).sum()
        )

    # Calculate the maximum lag and cumulation to determine the starting row for slicing.
    max_lag = max(lag_days) if lag_days else 0
    max_cumulation = max(cumulation_days) if cumulation_days else 0
    start_row = max(max_lag, max_cumulation)

    # Slice the DataFrame to remove NaN values from lagging and cumulative features.
    new_dataset = new_dataset[start_row:]

    return new_dataset
