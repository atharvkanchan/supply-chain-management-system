import streamlit as st
import pandas as pd
import plotly.express as px

# ---------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------
st.set_page_config(
    page_title="Supply Chain Analytics Dashboard",
    page_icon="📦",
    layout="wide"
)

# ---------------------------------------------
# HEADER
# ---------------------------------------------
st.title("📦 Supply Chain Data Analytics Dashboard")
st.markdown("An interactive dashboard for analyzing key supply chain performance metrics and trends.")

# ---------------------------------------------
# SIDEBAR
# ---------------------------------------------
st.sidebar.header("🔧 Dashboard Controls")

uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("supply_chain_data.csv")

# ---------------------------------------------
# DATA PREVIEW
# ---------------------------------------------
st.subheader("📋 Dataset Overview")
st.write("Preview of the uploaded supply chain data:")
st.dataframe(df.head())

# Basic data info
st.write("**Dataset Shape:**", df.shape)
st.write("**Missing Values:**")
st.dataframe(df.isnull().sum())

# ---------------------------------------------
# KPI METRICS
# ---------------------------------------------
st.subheader("📊 Key Performance Indicators")

col1, col2, col3 = st.columns(3)

with col1:
    total_cost = df["Cost"].sum() if "Cost" in df.columns else 0
    st.metric("Total Cost", f"${total_cost:,.0f}")

with col2:
    avg_lead_time = df["Lead Time"].mean() if "Lead Time" in df.columns else 0
    st.metric("Average Lead Time", f"{avg_lead_time:.2f} days")

with col3:
    total_orders = len(df)
    st.metric("Total Orders", total_orders)

# ---------------------------------------------
# FILTERS
# ---------------------------------------------
st.sidebar.subheader("📁 Filter Options")

if "Region" in df.columns:
    region_filter = st.sidebar.multiselect("Select Region(s)", df["Region"].unique(), default=df["Region"].unique())
    df = df[df["Region"].isin(region_filter)]

if "Product Type" in df.columns:
    product_filter = st.sidebar.multiselect("Select Product Type(s)", df["Product Type"].unique(), default=df["Product Type"].unique())
    df = df[df["Product Type"].isin(product_filter)]

# ---------------------------------------------
# VISUALIZATIONS
# ---------------------------------------------
st.subheader("📈 Data Visualizations")

tab1, tab2, tab3 = st.tabs(["Cost Analysis", "Regional Performance", "Time Trends"])

# 1️⃣ COST ANALYSIS
with tab1:
    if "Product Type" in df.columns and "Cost" in df.columns:
        cost_chart = px.bar(df, x="Product Type", y="Cost", color="Product Type",
                            title="Cost by Product Type", text_auto=True)
        st.plotly_chart(cost_chart, use_container_width=True)

# 2️⃣ REGIONAL PERFORMANCE
with tab2:
    if "Region" in df.columns and "Revenue" in df.columns:
        region_chart = px.pie(df, names="Region", values="Revenue", title="Revenue Distribution by Region")
        st.plotly_chart(region_chart, use_container_width=True)

# 3️⃣ TIME TRENDS
with tab3:
    date_columns = [col for col in df.columns if "Date" in col or "date" in col]
    if date_columns:
        df[date_columns[0]] = pd.to_datetime(df[date_columns[0]], errors='coerce')
        time_chart = px.line(df.sort_values(by=date_columns[0]),
                             x=date_columns[0], y="Revenue",
                             title="Revenue Trend Over Time")
        st.plotly_chart(time_chart, use_container_width=True)

# ---------------------------------------------
# INSIGHTS
# ---------------------------------------------
st.subheader("💡 Insights Summary")
st.markdown("""
- **High-cost products** can be identified in the cost analysis tab.  
- **Top-performing regions** can be found in the regional performance tab.  
- **Time-based trends** help identify seasonal demand and cost fluctuations.  
""")

# ---------------------------------------------
# FOOTER
# ---------------------------------------------
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit & Plotly | Intermediate Professional Dashboard")
