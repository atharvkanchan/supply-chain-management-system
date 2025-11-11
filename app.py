# app.py

import streamlit as st
import pandas as pd
import plotly.express as px

# ---------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------
st.set_page_config(
    page_title="Fashion Supply Management Analytics",
    page_icon="👗",
    layout="wide"
)

# ---------------------------------------------
# HEADER
# ---------------------------------------------
st.title("👗 Fashion Supply Chain Analytics Dashboard")
st.markdown("Gain insights into supply performance, product revenue, and operational efficiency.")

st.sidebar.header("📂 Upload & Filters")

# ---------------------------------------------
# DATA UPLOAD
# ---------------------------------------------
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = [c.strip().title() for c in df.columns]  # Normalize column names

    st.success("✅ Data uploaded successfully!")
    st.dataframe(df.head())

    # ---------------------------------------------
    # HANDLE EXPECTED COLUMNS
    # ---------------------------------------------
    expected_columns = ['Product Type', 'Region', 'Cost', 'Revenue', 'Lead Time', 'Date']
    missing_cols = [col for col in expected_columns if col not in df.columns]

    if missing_cols:
        st.warning(f"⚠️ Missing columns in dataset: {missing_cols}")

    # Convert date and numeric fields
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    numeric_cols = ['Cost', 'Revenue', 'Lead Time']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # ---------------------------------------------
    # SIDEBAR FILTERS
    # ---------------------------------------------
    st.sidebar.subheader("🔍 Filter Data")

    if 'Region' in df.columns:
        region_list = df['Region'].dropna().unique().tolist()
        region_select = st.sidebar.multiselect("Select Region(s)", region_list, default=region_list)
        df = df[df['Region'].isin(region_select)]

    if 'Product Type' in df.columns:
        product_list = df['Product Type'].dropna().unique().tolist()
        product_select = st.sidebar.multiselect("Select Product Type(s)", product_list, default=product_list)
        df = df[df['Product Type'].isin(product_select)]

    if 'Date' in df.columns:
        min_date, max_date = df['Date'].min(), df['Date'].max()
        start_date, end_date = st.sidebar.date_input("Select Date Range", [min_date, max_date])
        df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]

    # ---------------------------------------------
    # KPI METRICS
    # ---------------------------------------------
    st.markdown("### 📈 Key Performance Indicators")

    col1, col2, col3 = st.columns(3)
    if 'Revenue' in df.columns:
        total_rev = df['Revenue'].sum()
        col1.metric("Total Revenue", f"₹{total_rev:,.0f}")

    if all(col in df.columns for col in ['Revenue', 'Cost']):
        total_profit = (df['Revenue'] - df['Cost']).sum()
        avg_profit = (df['Revenue'] - df['Cost']).mean()
        col2.metric("Total Profit", f"₹{total_profit:,.0f}")
        col3.metric("Avg Profit/Transaction", f"₹{avg_profit:,.0f}")

    if 'Lead Time' in df.columns:
        avg_lead = df['Lead Time'].mean()
        col3.metric("Avg Lead Time", f"{avg_lead:.1f} days")

    st.markdown("---")

    # ---------------------------------------------
    # VISUALIZATIONS
    # ---------------------------------------------
    st.subheader("📊 Visual Insights")

    # Revenue trend
    if 'Date' in df.columns and 'Revenue' in df.columns:
        st.markdown("#### 💹 Revenue Trend Over Time")
        fig1 = px.line(df, x='Date', y='Revenue', color='Product Type', title="Revenue Over Time")
        st.plotly_chart(fig1, use_container_width=True)

    # Revenue by region
    if all(col in df.columns for col in ['Region', 'Revenue']):
        st.markdown("#### 🌍 Revenue by Region")
        region_summary = df.groupby('Region', as_index=False)['Revenue'].sum()
        fig2 = px.bar(region_summary, x='Region', y='Revenue', color='Region', title="Total Revenue by Region")
        st.plotly_chart(fig2, use_container_width=True)

    # Profit by product type
    if all(col in df.columns for col in ['Product Type', 'Revenue', 'Cost']):
        st.markdown("#### 🏷️ Profit by Product Type")
        df['Profit'] = df['Revenue'] - df['Cost']
        profit_summary = df.groupby('Product Type', as_index=False)['Profit'].sum()
        fig3 = px.bar(profit_summary, x='Product Type', y='Profit', color='Product Type', title="Profit by Product Type")
        st.plotly_chart(fig3, use_container_width=True)

    # Lead time analysis
    if all(col in df.columns for col in ['Product Type', 'Lead Time']):
        st.markdown("#### ⏱️ Lead Time Distribution")
        fig4 = px.box(df, x='Product Type', y='Lead Time', color='Product Type', title="Lead Time by Product Type")
        st.plotly_chart(fig4, use_container_width=True)

    # ---------------------------------------------
    # INSIGHTS
    # ---------------------------------------------
    st.subheader("🔍 Insights Summary")
    if all(col in df.columns for col in ['Revenue', 'Cost']):
        high_perf_product = df.groupby('Product Type')['Profit'].sum().idxmax()
        low_perf_product = df.groupby('Product Type')['Profit'].sum().idxmin()
        st.success(f"🟢 Highest profit product type: **{high_perf_product}**")
        st.error(f"🔴 Lowest profit product type: **{low_perf_product}**")

    if 'Region' in df.columns:
        top_region = df.groupby('Region')['Revenue'].sum().idxmax()
        st.info(f"📍 Top performing region by revenue: **{top_region}**")

    st.markdown("---")
    st.caption("👗 Developed using Streamlit, Plotly, and Pandas | Inspired by the Fashion Supply Chain Management Dashboard")

else:
    st.info("📁 Upload a CSV file from your supply chain dataset to begin analysis.")


