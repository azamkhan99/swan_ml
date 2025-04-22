# SWAN-ML Project

## Overview
The SWAN-ML project focuses on analyzing and predicting water quality using machine learning techniques. This project integrates data preprocessing, exploratory data analysis (EDA), model training, and inference to provide insights into climate and water quality dynamics.

## Project Structure

### Notebooks
- **cal_swat_dataset_creation.ipynb**: Dataset creation for SWAT calibration.
- **cal_swat_eda_comb.ipynb**: Combined exploratory data analysis for SWAT calibration.
- **eda.ipynb**: General exploratory data analysis.
- **henrique_code.ipynb**: Custom analysis scripts.
- **other_site.ipynb**: Analysis for other sites.
- **simulation_extractions.ipynb**: Extracting simulation results.

### Data
Located in the `data/` folder, this includes:
- Climate data (e.g., `Monthly_Climate_Data-1993-2020.csv`)
- Water quality data (e.g., `CalibratedWaterQuality.xlsx`)
- Streamflow data (e.g., `Streamflow Outlet.xlsx`)

### Figures
Located in the `figs/` folder, this includes visualizations such as:
- Correlation heatmaps
- SWAT projections
- Target vs. climate comparisons

### Scripts
#### Future Climate Inference
- `climate_change_data_inference_ffnn.py`: Inference using feedforward neural networks.
- `climate_change_data_inference_residuals.py`: Residual-based inference.
- `climate_change_data_inference.py`: General inference script.

#### Model Tuning
- `NN_hp_tuning.py`: Hyperparameter tuning for neural networks.
- `residuals_hp_tuning.py`: Hyperparameter tuning for residual models.
- `tabnet_hp_tuning.py`: Hyperparameter tuning for TabNet models.

### Models
Located in the `models/` folder, this includes trained models such as:
- `ffnn_Nitrate_DlyLd(kg).pt`
- `residual_Nitrate_DlyLd(kg).pt`

### Preprocessed Data
Located in the `preprocessed_data/` folder, this includes preprocessed datasets for training and inference.

## Requirements
Install the required Python packages using:
```bash
pip install -r requirements.txt
```

## Usage
1. **Data Preparation**: Use the notebooks in the root directory to preprocess and analyze data.
2. **Model Training**: Use scripts in `model_tuning_scripts/` to train and tune models.
3. **Inference**: Use scripts in `future_climate_inference_scripts/` to perform inference on future climate data.

## Results
Results and analyses are stored in the `mlruns/` folder, organized by experiment IDs, and the `ml_prediction_outputs`.
