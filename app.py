# app.py

import streamlit as st
import pandas as pd
import plotly.express as px

# 📍 Page configuration
st.set_page_config(
    page_title="Fashion Supply-Management Analytics",
    page_icon="👗",
    layout="wide"
)

# Header
st.title("Fashion Supply Chain Analytics Dashboard")
st.markdown(
    """
    Explore product performance, regional insights, cost vs revenue,
    and lead time distribution for your fashion-supply dataset.
    """
)

# Sidebar — upload & filters
st.sidebar.header("Upload Data & Filters")

uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is None:
    st.info("📥 Please upload a CSV file to start the analysis.")
    st.stop()

# Load data
df = pd.read_csv(uploaded_file)
# Normalize column names
df.columns = [c.strip().title() for c in df.columns]

# Expected columns
expected = ['Product Type', 'Region', 'Cost', 'Revenue', 'Lead Time', 'Date']
missing = [c for c in expected if c not in df.columns]
if missing:
    st.warning(f"⚠️ Missing expected columns: {missing}")
    st.info("Some charts/metrics may not be available due to missing columns.")

# Convert types
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
for col in ['Cost', 'Revenue', 'Lead Time']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Sidebar filters
st.sidebar.subheader("Filter Options")

if 'Region' in df.columns:
    regions = df['Region'].dropna().unique().tolist()
    sel_regions = st.sidebar.multiselect("Select Region(s)", regions, default=regions)
    df = df[df['Region'].isin(sel_regions)]

if 'Product Type' in df.columns:
    product_types = df['Product Type'].dropna().unique().tolist()
    sel_products = st.sidebar.multiselect("Select Product Type(s)", product_types, default=product_types)
    df = df[df['Product Type'].isin(sel_products)]

if 'Date' in df.columns:
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    sel_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])
    if len(sel_range) == 2:
        start_date, end_date = sel_range
        df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]

# KPI Metrics
st.markdown("### Key Metrics")
col1, col2, col3 = st.columns(3)

if 'Revenue' in df.columns:
    total_revenue = df['Revenue'].sum()
    col1.metric("Total Revenue", f"₹{total_revenue:,.0f}")

if all(c in df.columns for c in ['Revenue', 'Cost']):
    total_profit = (df['Revenue'] - df['Cost']).sum()
    avg_profit = (df['Revenue'] - df['Cost']).mean()
    col2.metric("Total Profit", f"₹{total_profit:,.0f}")
    col3.metric("Avg Profit per Transaction", f"₹{avg_profit:,.0f}")

if 'Lead Time' in df.columns:
    avg_lead = df['Lead Time'].mean()
    col3.metric("Average Lead Time (days)", f"{avg_lead:.1f}")

st.markdown("---")

# Visualizations
st.subheader("Visual Insights")

# Revenue Trend Over Time
if 'Date' in df.columns and 'Revenue' in df.columns:
    st.markdown("#### Revenue Trend Over Time")
    fig_trend = px.line(df, x='Date', y='Revenue', color='Product Type' if 'Product Type' in df.columns else None,
                        title="Revenue Over Time by Product Type")
    st.plotly_chart(fig_trend, use_container_width=True)

# Revenue by Region
if 'Region' in df.columns and 'Revenue' in df.columns:
    st.markdown("#### Revenue by Region")
    region_sum = df.groupby('Region', as_index=False)['Revenue'].sum()
    fig_region = px.bar(region_sum, x='Region', y='Revenue', color='Region',
                        title="Revenue by Region")
    st.plotly_chart(fig_region, use_container_width=True)

# Profit by Product Type
if all(c in df.columns for c in ['Product Type', 'Revenue', 'Cost']):
    st.markdown("#### Profit by Product Type")
    df['Profit'] = df['Revenue'] - df['Cost']
    prod_profit = df.groupby('Product Type', as_index=False)['Profit'].sum()
    fig_prod = px.bar(prod_profit, x='Product Type', y='Profit', color='Product Type',
                      title="Profit by Product Type")
    st.plotly_chart(fig_prod, use_container_width=True)

# Lead Time Distribution
if all(c in df.columns for c in ['Product Type', 'Lead Time']):
    st.markdown("#### Lead Time Distribution by Product Type")
    fig_lead = px.box(df, x='Product Type', y='Lead Time', color='Product Type',
                      title="Lead Time by Product Type")
    st.plotly_chart(fig_lead, use_container_width=True)

# Data Preview
st.markdown("### Data Preview")
st.dataframe(df.head())

# Insights Summary
st.subheader("Insights Summary")
if all(c in df.columns for c in ['Revenue', 'Cost', 'Product Type']):
    highest = df.groupby('Product Type')['Profit'].sum().idxmax()
    lowest = df.groupby('Product Type')['Profit'].sum().idxmin()
    st.success(f"Highest profit product type: **{highest}**")
    st.error(f"Lowest profit product type: **{lowest}**")

if 'Region' in df.columns and 'Revenue' in df.columns:
    best_region = df.groupby('Region')['Revenue'].sum().idxmax()
    st.info(f"Top region by revenue: **{best_region}**")

st.markdown("---")
st.caption("Dashboard built with Streamlit | Inspired by Fashion Supply-Chain Analytics App")


