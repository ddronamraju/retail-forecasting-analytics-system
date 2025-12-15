# 🛒 Retail Forecasting & Decision Intelligence System

This repository contains a modular, production-inspired forecasting and analytics system
designed for large retailers (e.g., Walmart, Target, Amazon Retail).

## Components

- **Baseline Prophet Forecast** – Store–Department level baseline forecasting and seasonality decomposition.
- **Anomaly Detection** – Spike/dip identification for sales monitoring.
- **Price Elasticity & Promo Impact** – Modeling price sensitivity and uplift.
- **SKU Segmentation** – Clustering items by demand patterns.
- **Inventory Risk Modeling** – Safety stock and stockout probability.
- **Feature Engineering Toolkit** – Lags, rolling windows, Fourier features.
- **LightGBM Forecasting** – Global production-style forecasting model.
- **Scenario Simulator** – Price/promo impact tool for business users.
- **Retail KPI Dashboard** – Executive summary of key retail KPIs.

## ⚠️ Important Notes (For Credibility)

### 1. Price Elasticity Uses Simulated Data
The **Price Elasticity** module uses **simulated price signals** because the Walmart dataset does not provide actual pricing data. 

- In the scenario simulator app, this is clearly labeled: *"Elasticity demo using simulated price signal; replace with real price/promo history in production."*
- For production use, replace the simulation with real historical price and promotion data from your transaction systems.

### 2. LightGBM Model Scope
The **LightGBM Forecasting** module currently trains a **per Store–Dept model** for demonstration purposes.

- In production retail environments, you would typically train a **global model** across many Store–Dept combinations or SKU-Store IDs to leverage cross-learning and handle cold-start scenarios.
- This demo approach showcases the methodology, but scalability requires pooling data across multiple locations/products.

📄 For a detailed narrative, see **CASE_STUDY.md**.

# Running streamlit app
python run_all_notebooks.py
streamlit run modules/scenario_simulator/unified_app.py