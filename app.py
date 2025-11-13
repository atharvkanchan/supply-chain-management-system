import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Supply Chain Dashboard",
    layout="wide",
    page_icon="📦"
)

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    return pd.read_csv("india_supply_chain_2024_2025.csv")

df = load_data()

# -------------------- HEADER --------------------
st.markdown("""
    <h1 style='text-align:center; color:#4F46E5;'>📦 Supply Chain Analytics Dashboard</h1>
    <h4 style='text-align:center; color:gray;'>AI Powered Forecasting • Insights • Optimization</h4>
    <br>
""", unsafe_allow_html=True)

# -------------------- KPIs --------------------
st.subheader("📊 Key Metrics")
c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Records", f"{len(df):,}")
c2.metric("Unique Products", df["Product"].nunique() if "Product" in df else "N/A")
c3.metric("Avg Lead Time", round(df["Lead_Time"].mean(), 2) if "Lead_Time" in df else "N/A")
c4.metric("Avg Demand", round(df["Demand"].mean(), 2) if "Demand" in df else "N/A")

# -------------------- DATA PREVIEW --------------------
st.subheader("📁 Dataset Preview")
st.dataframe(df.head(15), use_container_width=True)

# -------------------- ANALYTICS TABS --------------------
st.subheader("📈 Visual Insights")
t1, t2, t3 = st.tabs(["📦 Demand Trend", "🏭 Supplier Analysis", "🔥 Correlation Heatmap"])

with t1:
    if "Date" in df.columns and "Demand" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df2 = df.sort_values("Date")

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df2["Date"], df2["Demand"], linewidth=2)
        ax.set_title("Demand Trend Over Time")
        st.pyplot(fig)
    else:
        st.warning("Date or Demand column missing.")

with t2:
    if "Supplier" in df.columns and "Quantity" in df.columns:
        supplier_stats = df.groupby("Supplier")["Quantity"].sum()

        fig, ax = plt.subplots(figsize=(12, 4))
        supplier_stats.plot(kind="bar", ax=ax)
        ax.set_title("Supplier Quantity Contribution")
        st.pyplot(fig)
    else:
        st.warning("Supplier or Quantity column missing.")

with t3:
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(df.corr(), annot=True, cmap="Blues")
    st.pyplot(fig)

# -------------------- PREDICTION ENGINE --------------------
st.subheader("🔮 AI Demand Prediction")

try:
    model = joblib.load("model.pkl")
    st.success("Model Loaded Successfully!")
except:
    st.error("ERROR: model.pkl not found. Train the model using train_model.py")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

inputs = {}
cols = st.columns(3)
i = 0

for col in numeric_cols:
    if col != "Demand":
        with cols[i % 3]:
            inputs[col] = st.number_input(
                f"{col}",
                float(df[col].min()),
                float(df[col].max()),
                float(df[col].mean())
            )
        i += 1

if st.button("Predict Demand"):
    inp = pd.DataFrame([inputs])
    pred = model.predict(inp)[0]
    st.success(f"📌 Predicted Demand: **{round(pred,2)} units**")

# -------------------- FOOTER --------------------
st.markdown("""
<br><hr>
<p style='text-align:center; color:gray;'>
Developed by <b>Atharv Kanchan</b> • AI-based Supply Chain System 🚀
</p>
""", unsafe_allow_html=True)
