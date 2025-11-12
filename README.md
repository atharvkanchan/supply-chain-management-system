# 🇮🇳 India Supply Chain Management Dashboard (Streamlit)

This Streamlit dashboard provides **data analytics**, **ML model training**, and **prediction capabilities** for the India Supply Chain dataset (2024–2025).

---

## 🚀 Features
- Interactive data analytics and visualization
- Automatic feature preprocessing and model training (RandomForest)
- Single-row and batch prediction modes
- Built-in feature importance chart
- Support for custom CSV uploads and retraining inside the app

---

## 🧠 Model Overview
The included `model.pkl` is a trained **RandomForestRegressor** pipeline.  
It predicts `actual_sales` using multiple supply chain features such as:
- Distance, demand forecast, units shipped, cost per unit, GST, carbon, etc.
- Categorical features like city, transport mode, supplier, and carrier.

Model metrics (on test data):
- **MAE:** 1.7496  
- **MSE:** 4.6492  
- **R²:** 0.9542  

---

## 🧰 Requirements
Install all dependencies using:
```bash
pip install -r requirements.txt
