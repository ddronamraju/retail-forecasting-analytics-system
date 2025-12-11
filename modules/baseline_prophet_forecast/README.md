# Baseline Prophet Forecast (Store–Department Level)

## Objective

Establish an interpretable, store–department level baseline forecast using Prophet.  
This serves as the foundational component of the broader retail forecasting and decision system.

## Functionality

- Loads Walmart-like sales data: `train.csv`, `features.csv`, `stores.csv`.
- Filters to a specific `Store`–`Dept` combination.
- Fits a Prophet model with weekly and yearly seasonality.
- Generates a 12-week ahead forecast.
- Produces:
  - `images/forecast_store1_dept1.png`
  - `images/components_store1_dept1.png`
  - `baseline_prophet_forecast.ipynb`

## Role in the Overall System

- Provides a **baseline forecast** for comparison with the global LightGBM model.
- Offers **trend and seasonality decomposition** that can be shared directly with stakeholders.
- Acts as a **monitoring reference** in production to detect drift or unexpected behavior in more complex models.

## Production & Scaling Notes

- Prophet is well-suited for **aggregated series** (e.g., store-level, department-level, region-level) rather than millions of individual SKUs.
- In a production environment, this baseline component would typically:
  - Run on a schedule (e.g., nightly or weekly) using a workflow tool such as Airflow or Dagster.
  - Persist forecasts to a data warehouse (e.g., Snowflake, BigQuery).
  - Feed monitoring dashboards that benchmark more complex models and highlight deviations.

## Files

- `baseline_prophet_forecast.ipynb`
- `data/train.csv`
- `data/features.csv`
- `data/stores.csv`
- `images/forecast_store1_dept1.png`
- `images/components_store1_dept1.png`