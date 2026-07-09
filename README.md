Week 2 — Steel Industry Energy Consumption: EDA, Feature Engineering & Baseline Modeling
Internship task for ItSimpleRa (Week 2). This repo covers a full mini machine-learning
workflow on the Steel Industry Energy Consumption Dataset (UCI):
exploratory data analysis and feature engineering, followed by baseline regression modeling.
Contents
`week2_eda.ipynb` — Part 1: deep EDA, datetime feature extraction, `Power_Factor_Ratio` and
`High_Load` feature engineering, IQR outlier detection, correlation analysis, and visualizations
by load type and hour of day.
`week2_baseline_models.ipynb` — Part 2: encodes engineered features, trains Linear Regression,
Ridge Regression, Decision Tree, and Random Forest regressors, evaluates with MAE/RMSE/R² and
5-fold cross-validation, and selects a baseline model to carry forward.
`data/` — holds the downloaded/extracted dataset (created automatically when the first notebook
cell runs).
`requirements.txt` — Python dependencies.
How to run
Open either notebook in Google Colab or Jupyter.
Run all cells top to bottom — the first cell in each notebook downloads and extracts the
dataset into `data/` automatically.
Plots are saved as `.png` files alongside the notebook for easy embedding in a LinkedIn post or
report.
Dataset
Name: Steel Industry Energy Consumption Dataset
Source: UCI Machine Learning Repository
Link: https://archive.ics.uci.edu/static/public/851/steel+industry+energy+consumption.zip
Description: Real energy consumption readings from a steel manufacturing plant, including
usage (kWh), reactive power, power factors, CO2 output, load type, and timestamps.
Key design decisions
Categorical encoding: One-hot encoding was used for `Load_Type`, day-of-week, and month
features rather than label encoding, since these are nominal (unordered) categories — label
encoding would incorrectly imply a numeric ordering between categories.
Leakage avoidance: `High_Load` (derived directly from `Usage_kWh`, the modeling target) is
dropped before training to avoid leaking target information into the model.
Reproducibility: All train/test splits and models use `random_state=42`.
Author
Taha — ItSimpleRa Data Analytics Internship, Week 2
