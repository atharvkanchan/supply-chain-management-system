import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Supply Chain Dashboard", layout="wide")

st.title("Supply Chain Management — Dashboard & Predict")
st.markdown("""
Upload your dataset (CSV) or a trained model (`model.pkl`).
This app shows: exploratory analytics, model evaluation, and single/batch prediction.
""")

# Sidebar - model upload or load default
st.sidebar.header("Model / Data")
uploaded_model = st.sidebar.file_uploader("Upload trained model (`.pkl`)", type=["pkl","joblib"])
uploaded_csv = st.sidebar.file_uploader("Upload dataset (CSV) for dashboard / training", type=["csv"])

@st.cache_data
def load_model_from_file(f):
    return joblib.load(f)

model = None
if uploaded_model is not None:
    try:
        model = load_model_from_file(uploaded_model)
        st.sidebar.success("Model loaded from uploaded file.")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")

# If no model uploaded, check for model.pkl in root
if model is None:
    try:
        model = joblib.load("model.pkl")
        st.sidebar.info("Loaded model.pkl from app root.")
    except Exception:
        model = None

df = None
if uploaded_csv is not None:
    df = pd.read_csv(uploaded_csv)
    st.sidebar.success("CSV loaded")

st.header("1) Dataset preview")
if df is None:
    st.info("No CSV uploaded. You can upload a dataset to see analytics and retrain the model.")
else:
    st.dataframe(df.head())
    st.markdown(f"**Shape:** {df.shape}")

    st.subheader("Quick EDA")
    numeric = df.select_dtypes(include=["int","float"])
    st.write("Numeric columns:", list(numeric.columns))
    col = st.selectbox("Choose numeric column to plot distribution", options=numeric.columns)
    fig, ax = plt.subplots()
    ax.hist(df[col].dropna(), bins=30)
    ax.set_title(f"Distribution of {col}")
    st.pyplot(fig)

st.header("2) Train / Improve model (optional)")
st.markdown("""
If you upload a CSV and it contains a target column named `target`, the app will:
- Clean data
- Train a RandomForestRegressor
- Save `model.pkl` for predictions
""")

if df is not None and "target" in df.columns:
    if st.button("Train improved model (RandomForest)"):
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.compose import ColumnTransformer

        X = df.drop(columns=["target"])
        y = df["target"].values

        numeric_cols = X.select_dtypes(include=["int","float"]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object","category"]).columns.tolist()

        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        from sklearn.preprocessing import OneHotEncoder
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))
        ])

        preproc = ColumnTransformer([
            ("num", numeric_pipe, numeric_cols),
            ("cat", cat_pipe, cat_cols)
        ], remainder="drop")

        pipe = Pipeline([
            ("preproc", preproc),
            ("model", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
        ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        joblib.dump(pipe, "model.pkl")
        st.success("Training complete — saved model.pkl")
        st.write("MAE:", round(mae,4), "MSE:", round(mse,4), "R2:", round(r2,4))
else:
    st.info("Upload a CSV containing a `target` column to enable training.")

st.header("3) Predict")
st.markdown("Use the loaded model (or upload a model) to make single-row or batch predictions.")

if model is None:
    st.warning("No model available. Upload `model.pkl` or train a model with a dataset containing `target`.")
else:
    st.subheader("Single prediction")
    st.write("Enter values for features (features are inferred automatically if sample CSV uploaded).")
    st.subheader("Batch prediction (CSV)")
    batch_file = st.file_uploader("Upload CSV for batch prediction", type=["csv"], key="batch")
    if batch_file is not None:
        batch_df = pd.read_csv(batch_file)
        preds = model.predict(batch_df)
        batch_df["prediction"] = preds
        st.dataframe(batch_df.head())
        csv = batch_df.to_csv(index=False).encode()
        st.download_button("Download predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")

st.markdown("---")
st.write("Built with ❤️ by Atharv Kanchan — India Supply Chain Dashboard (2024–2025)")
