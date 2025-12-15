# LightGBM Forecasting (Production-Style)

## Objective

Train a production-inspired LightGBM forecasting model using leakage-safe engineered features.
This module demonstrates how retail demand forecasting is commonly implemented at scale using
gradient boosting models.

## Functionality

- Loads weekly sales data for a Store–Department series.
- Generates features using the shared feature engineering toolkit:
  - lags, rolling stats, YoY change, holiday windows, Fourier terms
- Uses a time-based holdout split (last 12 weeks as validation).
- Trains a LightGBM regressor and evaluates with:
  - MAE
  - WAPE
- Produces:
  - Forecast vs actual plot
  - Feature importance plot
- Saves a model artifact for future serving.

## Outputs

- `images/forecast_vs_actual_store1_dept1.png`
- `images/feature_importance_top20.png`
- `lgbm_model_store1_dept1.pkl`
- Notebook: `lgbm_forecast.ipynb`

## Role in the Overall System

- Primary forecasting engine for downstream decision modules:
  - inventory risk (uncertainty via residuals)
  - scenario simulation inputs
  - KPI dashboards (forecast vs actual / bias tracking)

## Production & Scaling Notes

- Production retail forecasting typically uses **global models** trained across many items/locations.
- Batch inference runs nightly using orchestration (Airflow/Dagster) and distributed compute (Dask/Spark).
- Models are versioned in a registry (e.g., MLflow) and monitored for drift and bias over time.