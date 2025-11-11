# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Supply Chain Analytics Dashboard",
    page_icon="📦",
    layout="wide"
)

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.title("📦 Supply Chain Data Analytics Dashboard")
st.markdown("""
This dashboard provides **deep insights** into supply chain operations — including cost, revenue, 
profitability, and forecasting — using your dataset: **`supply_chain_data.csv`**
""")

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
st.sidebar.header("📁 Upload or Use Default Dataset")

uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
else:
    df = pd.read_csv("supply_chain_data.csv")
    st.info("Using default dataset: `supply_chain_data.csv`")

# Ensure proper column names
expected_columns = ["Product Type", "Region", "Cost", "Revenue", "Lead Time", "Date"]
missing_cols = [c for c in expected_columns if c not in df.columns]

if missing_cols:
    st.error(f"⚠️ Missing expected columns in dataset: {missing_cols}")
    st.stop()

# ---------------------------------------------------------
# DATA OVERVIEW
# ---------------------------------------------------------
st.subheader("📊 Dataset Overview")
st.dataframe(df.head(10))
st.write("Shape:", df.shape)
st.write("Missing Values:", df.isnull().sum().sum())

# ---------------------------------------------------------
# KPI METRICS
# ---------------------------------------------------------
st.subheader("📈 Key Metrics")

col1, col2, col3 = st.columns(3)
col1.metric("Total Revenue", f"${df['Revenue'].sum():,.0f}")
col2.metric("Total Cost", f"${df['Cost'].sum():,.0f}")
col3.metric("Average Lead Time", f"{df['Lead Time'].mean():.2f} days")

# ---------------------------------------------------------
# FILTERS
# ---------------------------------------------------------
st.sidebar.header("🔍 Data Filters")

regions = st.sidebar.multiselect("Select Region", df["Region"].unique(), df["Region"].unique())
products = st.sidebar.multiselect("Select Product Type", df["Product Type"].unique(), df["Product Type"].unique())

df = df[(df["Region"].isin(regions)) & (df["Product Type"].isin(products))]

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview", "Correlation Heatmap", "Profitability", "Outlier Detection", "Feature Importance", "Forecasting"
])

# ---------------------------------------------------------
# TAB 1: OVERVIEW
# ---------------------------------------------------------
with tab1:
    st.write("### Revenue and Cost by Product Type")
    fig1 = px.bar(df, x="Product Type", y=["Revenue", "Cost"], barmode="group", title="Revenue vs Cost by Product Type")
    st.plotly_chart(fig1, use_container_width=True)

    st.write("### Revenue by Region")
    fig2 = px.pie(df, names="Region", values="Revenue", title="Revenue Distribution by Region")
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------
# TAB 2: CORRELATION HEATMAP
# ---------------------------------------------------------
with tab2:
    st.write("### Correlation Analysis")
    num_df = df.select_dtypes(include=np.number)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ---------------------------------------------------------
# TAB 3: PROFITABILITY
# ---------------------------------------------------------
with tab3:
    st.write("### Profitability Analysis")
    df["Profit"] = df["Revenue"] - df["Cost"]
    profit_summary = df.groupby("Product Type")[["Revenue", "Cost", "Profit"]].sum().sort_values("Profit", ascending=False)
    st.dataframe(profit_summary)

    fig3 = px.bar(profit_summary, x=profit_summary.index, y="Profit", color="Profit", title="Profit by Product Type")
    st.plotly_chart(fig3, use_container_width=True)

# ---------------------------------------------------------
# TAB 4: OUTLIER DETECTION
# ---------------------------------------------------------
with tab4:
    st.write("### Outlier Detection")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    selected_col = st.selectbox("Select numeric column for analysis", numeric_cols)
    
    Q1 = df[selected_col].quantile(0.25)
    Q3 = df[selected_col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers = df[(df[selected_col] < lower) | (df[selected_col] > upper)]
    
    st.write(f"Detected **{len(outliers)}** outliers in `{selected_col}` column")
    fig4 = px.box(df, y=selected_col, points="all", title=f"Outlier Detection for {selected_col}")
    st.plotly_chart(fig4, use_container_width=True)

# ---------------------------------------------------------
# TAB 5: FEATURE IMPORTANCE
# ---------------------------------------------------------
with tab5:
    st.write("### Feature Importance (Random Forest Regressor)")
    df_model = df.select_dtypes(include=np.number).dropna()
    if "Revenue" in df_model.columns and len(df_model.columns) > 1:
        X = df_model.drop(columns=["Revenue"])
        y = df_model["Revenue"]
        model = RandomForestRegressor(random_state=42)
        model.fit(X, y)
        importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        st.bar_chart(importance)
        st.dataframe(importance.rename("Importance"))
    else:
        st.warning("Insufficient numeric data for feature importance analysis.")

# ---------------------------------------------------------
# TAB 6: FORECASTING
# ---------------------------------------------------------
with tab6:
    st.write("### Revenue Forecasting (Linear Regression)")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Revenue"]).sort_values("Date")
    df["Days"] = (df["Date"] - df["Date"].min()).dt.days

    X = df[["Days"]]
    y = df["Revenue"]

    model = LinearRegression()
    model.fit(X, y)

    future_days = np.arange(X["Days"].max() + 1, X["Days"].max() + 8).reshape(-1, 1)
    future_pred = model.predict(future_days)
    future_dates = pd.date_range(df["Date"].max(), periods=8, freq="D")[1:]

    fig5, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["Date"], y, label="Historical Revenue", marker="o")
    ax.plot(future_dates, future_pred, "--", color="orange", label="Forecasted Revenue")
    ax.legend()
    ax.set_title("Revenue Forecast for Next 7 Days")
    st.pyplot(fig5)

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("---")
st.caption("📊 Developed by Aniket Dombale | Advanced Streamlit Dashboard for Supply Chain Data Analytics")

