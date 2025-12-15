# LightGBM Forecasting (Production-Style)

## Objective

Train production-inspired LightGBM forecasting models using leakage-safe engineered features.
This module demonstrates how retail demand forecasting is commonly implemented at scale using
gradient boosting models.

## Two Approaches

### 1. Per-Entity Model (`lgbm_forecast.ipynb`)
- Trains one model for a single Store–Dept combination
- Best for: High-value SKUs with rich historical data
- Limitations: Cannot handle cold-start scenarios

### 2. Global Model (`global_lgbm_forecast.ipynb`) ⭐ **Recommended for Production**
- Trains one model across multiple Store–Dept combinations
- Leverages cross-learning from similar patterns
- Handles cold-start scenarios (new stores/products with zero history)
- Scalable to thousands of entities with single model

## Functionality (Per-Entity Model)

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

**Per-Entity Model:**
- `images/forecast_vs_actual_store1_dept1.png`
- `images/feature_importance_top20.png`
- `../../artifacts/lgbm_model_store1_dept1.pkl`

**Global Model:**
- `images/global_feature_importance.png`
- `images/global_model_predictions.png`
- `images/cold_start_prediction.png`
- `../../artifacts/global_lgbm_model.pkl`
- `../../artifacts/global_model_features.txt`

## Role in the Overall System

- Primary forecasting engine for downstream decision modules:
  - inventory risk (uncertainty via residuals)
  - scenario simulation inputs
  - KPI dashboards (forecast vs actual / bias tracking)

## Production & Scaling Notes

- **Global models are the industry standard** for large-scale retail forecasting (Amazon, Walmart, Target)
- Single global model can serve forecasts for 100k+ SKU-Store combinations
- Cold-start handling is critical for:
  - New product launches
  - Store openings/expansions
  - Seasonal merchandise with limited history
- Batch inference runs nightly using orchestration (Airflow/Dagster) and distributed compute (Dask/Spark)
- Models are versioned in a registry (e.g., MLflow) and monitored for drift and bias over time

## Key Differences: Per-Entity vs Global

| Aspect | Per-Entity | Global |
|--------|-----------|--------|
| Training | One model per Store-Dept | One model for all |
| Cold-start | ❌ Cannot predict new entities | ✅ Handles new entities |
| Maintenance | Manage 1000s of models | Manage 1 model |
| Cross-learning | None | Learns from all entities |
| Use Case | High-value SKUs | Scalable production |