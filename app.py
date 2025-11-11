# =====================================================
# app.py — Fashion Supply Chain Analytics Dashboard
# Dataset: supply_chain_data.csv
# Built with Streamlit + Plotly + Pandas
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ---------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------
st.set_page_config(
    page_title="Fashion Supply Chain Analytics",
    page_icon="🧵",
    layout="wide",
)

# ---------------------------------------
# LOAD DATA
# ---------------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("supply_chain_data.csv")
    return data

df = load_data()

# ---------------------------------------
# PAGE HEADER
# ---------------------------------------
st.title("👗 Fashion Supply Chain Analytics Dashboard")
st.markdown("""
#### Gain insights into production efficiency, logistics, and product performance.  
Use the filters on the sidebar to explore different suppliers, regions, and categories.
""")
st.markdown("---")

# ---------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------
st.sidebar.header("🔍 Filters")

product_types = df["Product type"].dropna().unique()
locations = df["Location"].dropna().unique()
suppliers = df["Supplier name"].dropna().unique()
transport_modes = df["Transportation modes"].dropna().unique()

selected_product = st.sidebar.selectbox("Select Product Type", product_types)
selected_location = st.sidebar.selectbox("Select Location", locations)
selected_supplier = st.sidebar.selectbox("Select Supplier", suppliers)
selected_transport = st.sidebar.selectbox("Select Transport Mode", transport_modes)

# Apply filters
filtered_df = df[
    (df["Product type"] == selected_product)
    & (df["Location"] == selected_location)
    & (df["Supplier name"] == selected_supplier)
    & (df["Transportation modes"] == selected_transport)
]

# ---------------------------------------
# KPI SECTION
# ---------------------------------------
st.subheader("📊 Key Performance Indicators")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Revenue ($)", f"{filtered_df['Revenue generated'].sum():,.2f}")
col2.metric("Products Sold", f"{filtered_df['Number of products sold'].sum():,}")
col3.metric("Avg Lead Time (Days)", f"{filtered_df['Lead times'].mean():.1f}")
col4.metric("Avg Manufacturing Cost ($)", f"{filtered_df['Manufacturing costs'].mean():.1f}")
col5.metric("Avg Defect Rate (%)", f"{filtered_df['Defect rates'].mean():.2f}")

st.markdown("---")

# ---------------------------------------
# TABS LAYOUT
# ---------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "💰 Sales & Performance",
    "🏭 Production & Quality",
    "🚚 Logistics & Cost",
    "⚠️ Risk & Supplier Analysis"
])

# ---------------------------------------
# TAB 1 – SALES & PERFORMANCE
# ---------------------------------------
with tab1:
    st.subheader("💰 Sales Performance Overview")

    colA, colB = st.columns(2)

    with colA:
        fig1 = px.bar(
            filtered_df,
            x="SKU",
            y="Revenue generated",
            color="Product type",
            title="Revenue by SKU",
            labels={"Revenue generated": "Revenue ($)"}
        )
        st.plotly_chart(fig1, use_container_width=True)

    with colB:
        fig2 = px.scatter(
            filtered_df,
            x="Price",
            y="Number of products sold",
            size="Revenue generated",
            color="Product type",
            hover_name="SKU",
            title="Price vs Units Sold (Bubble = Revenue)"
        )
        st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------
# TAB 2 – PRODUCTION & QUALITY
# ---------------------------------------
with tab2:
    st.subheader("🏭 Production Efficiency & Quality Metrics")

    colC, colD = st.columns(2)

    with colC:
        fig3 = px.box(
            filtered_df,
            x="Product type",
            y="Manufacturing costs",
            color="Product type",
            title="Manufacturing Cost Distribution by Product Type"
        )
        st.plotly_chart(fig3, use_container_width=True)

    with colD:
        fig4 = px.scatter(
            filtered_df,
            x="Manufacturing lead time",
            y="Defect rates",
            color="Inspection results",
            size="Production volumes",
            title="Defect Rates vs Manufacturing Lead Time"
        )
        st.plotly_chart(fig4, use_container_width=True)

# ---------------------------------------
# TAB 3 – LOGISTICS & COST
# ---------------------------------------
with tab3:
    st.subheader("🚚 Logistics & Transportation Costs")

    colE, colF = st.columns(2)

    with colE:
        fig5 = px.bar(
            filtered_df,
            x="Transportation modes",
            y="Shipping costs",
            color="Routes",
            title="Shipping Costs by Transport Mode"
        )
        st.plotly_chart(fig5, use_container_width=True)

    with colF:
        fig6 = px.box(
            filtered_df,
            x="Location",
            y="Costs",
            color="Transportation modes",
            title="Total Operational Costs by Location"
        )
        st.plotly_chart(fig6, use_container_width=True)

# ---------------------------------------
# TAB 4 – RISK & SUPPLIER ANALYSIS
# ---------------------------------------
with tab4:
    st.subheader("⚠️ Supplier Risk & Lead Time Analysis")

    risk_pivot = filtered_df.pivot_table(
        index="Supplier name",
        columns="Location",
        values="Defect rates",
        aggfunc="mean"
    )

    st.write("### 🧾 Average Defect Rate by Supplier & Location")
    st.dataframe(risk_pivot.style.background_gradient(cmap="Reds"))

    st.markdown("### 📉 Lead Time vs Defect Rate (Risk Plot)")
    fig7 = px.scatter(
        filtered_df,
        x="Lead times",
        y="Defect rates",
        color="Supplier name",
        size="Manufacturing costs",
        title="Lead Time vs Defect Rate per Supplier"
    )
    st.plotly_chart(fig7, use_container_width=True)

# ---------------------------------------
# DOWNLOAD SECTION
# ---------------------------------------
st.markdown("---")
st.subheader("📥 Download Filtered Data")

csv = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download CSV Data",
    data=csv,
    file_name="filtered_supply_chain_data.csv",
    mime="text/csv",
)

# ---------------------------------------
# FOOTER
# ---------------------------------------
st.markdown("---")
st.markdown("🧵 **Fashion Supply Chain Analytics Dashboard — Enhanced Edition**")
st.caption("© 2025 AnalyticsForFashionSupplyManagement | Developed with Streamlit & Plotly")







