# Sales Anomaly Detection (Store–Department Level)

## Objective

Detect statistically significant spikes and dips in weekly sales for a given store–department
time series. This module supports monitoring, alerting, and root cause analysis in a retail
forecasting and operations environment.

## Functionality

- Loads Walmart-like sales data (reused from the baseline Prophet module).
- Aggregates weekly sales at the Store–Dept level.
- Computes:
  - Rolling mean and standard deviation (8-week window).
  - Z-scores for each week.
- Flags anomalies where |z-score| exceeds a configurable threshold.
- Outputs:
  - Time series plot with anomalies highlighted.
  - A table of anomalous weeks with z-scores and holiday flags.

## Files

- `anomaly_detection.ipynb`
- `images/anomaly_series_store1_dept1.png`
- `anomalies_store1_dept1.csv` (optional, if saved)

## Role in the Overall System

- Acts as the **monitoring and alerting** layer for demand behavior.
- Helps distinguish between:
  - Expected spikes (holidays, planned promotions).
  - Unexpected anomalies (stockouts, supply chain issues, store closures).
- Can be extended to:
  - Run across all store–department combinations.
  - Feed anomaly summaries into dashboards for replenishment and merchandising teams.

## Production & Scaling Notes

- At scale, anomaly detection is run **daily or weekly** across thousands of time series.
- Implementation considerations:
  - Use **distributed compute** (Dask, Spark) to calculate rolling statistics and z-scores in parallel.
  - Store anomaly flags in a warehouse (e.g., Snowflake, BigQuery) for consumption by BI tools and alerting systems.
  - Integrate with forecasting error metrics (e.g., forecast residuals) to prioritize anomalies with the largest operational impact.
- Thresholds (e.g., |z| > 2) can be tuned per category or store cluster based on historical behavior.