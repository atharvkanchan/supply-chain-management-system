# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# ---------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------
st.set_page_config(
    page_title="Fashion Supply Chain Analytics Dashboard",
    page_icon="👗",
    layout="wide"
)

# ---------------------------------------------
# HEADER SECTION
# ---------------------------------------------
st.title("👗 Fashion Supply Chain Analytics Dashboard")
st.markdown("""
A powerful analytics interface to explore revenue, cost, logistics, and production performance 
across the fashion supply network.
""")

st.sidebar.header("📂 Upload Data & Filters")

# ---------------------------------------------
# FILE UPLOAD
# ---------------------------------------------
uploaded_file = st.sidebar.file_uploader("Upload Supply Chain Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = [c.strip().title() for c in df.columns]  # Normalize columns
    st.success("✅ Dataset successfully uploaded!")
else:
    st.info("📁 Please upload your CSV file to begin.")
    st.stop()

# ---------------------------------------------
# HANDLE EXPECTED COLUMNS
# ---------------------------------------------
expected_columns = [
    'Product', 'Location', 'Transport', 'Revenue', 'Cost', 
    'Stock Rate', 'Total Production', 'Defect Rate', 
    'Orders', 'Lead Time', 'Shipping Cost', 'Date'
]

missing_cols = [col for col in expected_columns if col not in df.columns]
if missing_cols:
    st.warning(f"⚠️ Missing expected columns in dataset: {missing_cols}")
    st.info("Some metrics may not appear due to missing data.")

# Convert column types
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

numeric_cols = ['Revenue', 'Cost', 'Stock Rate', 'Total Production', 
                'Defect Rate', 'Orders', 'Lead Time', 'Shipping Cost']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# ---------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------
st.sidebar.subheader("🔍 Filter Data")

if 'Product' in df.columns:
    product_options = df['Product'].dropna().unique().tolist()
    selected_products = st.sidebar.multiselect("Select Product(s)", product_options, default=product_options)
    df = df[df['Product'].isin(selected_products)]

if 'Location' in df.columns:
    location_options = df['Location'].dropna().unique().tolist()
    selected_locations = st.sidebar.multiselect("Select Location(s)", location_options, default=location_options)
    df = df[df['Location'].isin(selected_locations)]

if 'Transport' in df.columns:
    transport_options = df['Transport'].dropna().unique().tolist()
    selected_transport = st.sidebar.multiselect("Select Transport Mode(s)", transport_options, default=transport_options)
    df = df[df['Transport'].isin(selected_transport)]

# ---------------------------------------------
# KPI METRICS
# ---------------------------------------------
st.markdown("### 💠 Key Performance Metrics")

col1, col2, col3, col4 = st.columns(4)

if 'Revenue' in df.columns:
    col1.metric("Total Revenue", f"₹{df['Revenue'].sum():,.0f}")

if 'Cost' in df.columns:
    col2.metric("Total Cost", f"₹{df['Cost'].sum():,.0f}")

if 'Stock Rate' in df.columns:
    col3.metric("Average Stock Rate", f"{df['Stock Rate'].mean():.2f}%")

if 'Total Production' in df.columns:
    col4.metric("Total Production Units", f"{df['Total Production'].sum():,.0f}")

st.markdown("---")

# ---------------------------------------------
# KEY INSIGHTS SECTION
# ---------------------------------------------
st.subheader("📊 Key Insights Dashboard")

insight_col1, insight_col2, insight_col3 = st.columns(3)

if 'Revenue' in df.columns:
    revenue_perf = df['Revenue'].mean()
    insight_col1.success(f"💰 Revenue Performance\n\nAvg: ₹{revenue_perf:,.0f}")

if 'Orders' in df.columns:
    order_vol = df['Orders'].sum()
    insight_col2.info(f"📦 Order Volume\n\nTotal Orders: {order_vol:,}")

if all(c in df.columns for c in ['Cost', 'Revenue']):
    cost_eff = (1 - (df['Cost'].sum() / df['Revenue'].sum())) * 100
    insight_col3.success(f"💹 Cost Efficiency\n\n{cost_eff:.2f}%")

if 'Defect Rate' in df.columns:
    defect_ctrl = 100 - df['Defect Rate'].mean()
    insight_col1.warning(f"⚙️ Defect Rate Control\n\nEfficiency: {defect_ctrl:.2f}%")

if all(c in df.columns for c in ['Stock Rate', 'Orders']):
    stock_turn = df['Orders'].sum() / (df['Stock Rate'].mean() + 1)
    insight_col2.info(f"📈 Stock Turnover\n\nRate: {stock_turn:.2f}")

if all(c in df.columns for c in ['Shipping Cost', 'Revenue']):
    ship_eff = (df['Revenue'].sum() / df['Shipping Cost'].sum())
    insight_col3.success(f"🚚 Shipping Cost Efficiency\n\nRatio: {ship_eff:.2f}")

if 'Lead Time' in df.columns:
    lead_time = df['Lead Time'].mean()
    insight_col1.info(f"⏱️ Manufacturing Lead Time\n\nAvg: {lead_time:.2f} days")

st.markdown("---")

# ---------------------------------------------
# ANALYSIS SECTION
# ---------------------------------------------
st.subheader("📈 Detailed Analysis & Charts")

# Revenue Trend
if 'Date' in df.columns and 'Revenue' in df.columns:
    st.markdown("#### 💹 Revenue Trend Over Time")
    fig1 = px.line(df, x='Date', y='Revenue', color='Product', title="Revenue Over Time by Product")
    st.plotly_chart(fig1, use_container_width=True)

# Cost vs Revenue Comparison
if all(c in df.columns for c in ['Cost', 'Revenue', 'Product']):
    st.markdown("#### 💰 Cost vs Revenue by Product")
    fig2 = px.bar(df, x='Product', y=['Cost', 'Revenue'], barmode='group', title="Cost vs Revenue Comparison")
    st.plotly_chart(fig2, use_container_width=True)

# Location Performance
if all(c in df.columns for c in ['Location', 'Revenue']):
    st.markdown("#### 🌍 Location-Based Revenue Performance")
    loc_df = df.groupby('Location', as_index=False)['Revenue'].sum()
    fig3 = px.bar(loc_df, x='Location', y='Revenue', color='Location', title="Revenue by Location")
    st.plotly_chart(fig3, use_container_width=True)

# Transport Efficiency
if all(c in df.columns for c in ['Transport', 'Shipping Cost']):
    st.markdown("#### 🚛 Shipping Cost by Transport Mode")
    fig4 = px.pie(df, names='Transport', values='Shipping Cost', title="Shipping Cost Distribution")
    st.plotly_chart(fig4, use_container_width=True)

# Lead Time Distribution
if 'Lead Time' in df.columns and 'Product' in df.columns:
    st.markdown("#### ⏳ Lead Time Distribution by Product")
    fig5 = px.box(df, x='Product', y='Lead Time', color='Product', title="Lead Time Analysis")
    st.plotly_chart(fig5, use_container_width=True)

# Stock Rate vs Production
if all(c in df.columns for c in ['Stock Rate', 'Total Production']):
    st.markdown("#### 🧵 Stock Rate vs Total Production")
    fig6 = px.scatter(df, x='Stock Rate', y='Total Production', color='Product',
                      size='Revenue' if 'Revenue' in df.columns else None,
                      title="Stock Rate vs Production Volume")
    st.plotly_chart(fig6, use_container_width=True)

# Correlation Heatmap
if len(df.select_dtypes(include='number').columns) > 1:
    st.markdown("#### 📊 Correlation Heatmap")
    corr = df.select_dtypes(include='number').corr()
    fig7 = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation between Key Variables")
    st.plotly_chart(fig7, use_container_width=True)

st.markdown("---")

# ---------------------------------------------
# FOOTER
# ---------------------------------------------
st.markdown("""
🔗 **Reference Dashboard:** [Analytics for Fashion Supply Management](https://analyticsforfashionsupplymanagement.streamlit.app/)  
👗 *Developed with Streamlit, Plotly, and Pandas for Fashion Supply Chain Intelligence*
""")


