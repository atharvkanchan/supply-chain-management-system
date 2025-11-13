import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Supply Chain Dashboard",
    page_icon="📦",
    layout="wide"
)

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_dataset():
    return pd.read_csv("india_supply_chain_2024_2025.csv")

df = load_dataset()

# -------------------- HEADER --------------------
st.markdown(
    """
    <h1 style='text-align:center; color:#3B82F6;'>📦 Supply Chain Analytics Dashboard</h1>
    <h4 style='text-align:center; color:gray;'>AI Powered Inventory, Demand & Forecasting Insights</h4>
    <br>
    """,
    unsafe_allow_html=True
)

# -------------------- KPIs --------------------
st.subheader("📊 Key Metrics")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Records", f"{len(df):,}")
col2.metric("Unique Products", df['Product'].nunique() if 'Product' in df else "N/A")
col3.metric("Avg Lead Time", round(df['Lead_Time'].mean(), 2) if 'Lead_Time' in df else "N/A")
col4.metric("Avg Demand", round(df['Demand'].mean(), 2) if 'Demand' in df else "N/A")


# -------------------- DATA PREVIEW --------------------
st.subheader("📁 Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

# -------------------- CHARTS SECTION --------------------
st.subheader("📈 Visual Analytics")

tab1, tab2, tab3 = st.tabs(["📦 Demand Trends", "🏭 Supplier Analysis", "🔥 Heatmap"])

with tab1:
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        df_sorted = df.sort_values("Date")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df_sorted["Date"], df_sorted["Demand"], linewidth=2)
        ax.set_title("Demand Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Demand")
        st.pyplot(fig)
    else:
        st.warning("No 'Date' column found to plot demand trends.")

with tab2:
    if "Supplier" in df.columns and "Quantity" in df.columns:
        supplier_data = df.groupby("Supplier")["Quantity"].sum()

        fig, ax = plt.subplots(figsize=(10, 4))
        supplier_data.plot(kind="bar")
        ax.set_title("Supplier Quantity Contribution")
        ax.set_xlabel("Supplier")
        ax.set_ylabel("Total Quantity")
        st.pyplot(fig)
    else:
        st.warning("Supplier or Quantity column missing.")

with tab3:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.corr(), annot=True, cmap="Blues")
    st.pyplot(fig)


# -------------------- PREDICTION SECTION --------------------
st.subheader("🔮 AI Demand Prediction")
st.markdown("Enter input values to predict demand using the trained ML model.")

try:
    model = joblib.load("model.pkl")
    st.success("Model Loaded Successfully")
except:
    st.error("Model not found! Train the model first using train_model.py")

numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

input_values = {}
cols = st.columns(3)

i = 0
for col in numeric_columns:
    if col != "Demand":
        with cols[i % 3]:
            input_values[col] = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
        i += 1

if st.button("Predict Demand"):
    input_df = pd.DataFrame([input_values])
    prediction = model.predict(input_df)[0]
    st.success(f"📌 Predicted Demand: **{round(prediction,2)}** units")

# -------------------- FOOTER --------------------
st.markdown(
    """
    <br><hr>
    <p style='text-align:center; color:gray;'>
    Developed by <b>Atharv Kanchan</b> | AI-Enhanced Supply Chain System 🚀
    </p>
    """,
    unsafe_allow_html=True
)
