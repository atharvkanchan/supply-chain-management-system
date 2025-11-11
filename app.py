# ==========================================
# app.py
# Fashion Supply Chain Analytics Dashboard
# Built with Streamlit + Plotly
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# ---------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------
st.set_page_config(
    page_title="Fashion Supply Chain Analytics",
    page_icon="🧵",
    layout="wide",
)

# ---------------------------------------
# HEADER & INTRO
# ---------------------------------------
st.title("👗 Fashion Supply Chain Analytics Dashboard")
st.markdown("""
#### Visualize, monitor, and optimize fashion supply chain operations  
Track key metrics like lead time, order volume, delivery rate, and returns — all in one place.
""")

st.markdown("---")

# ---------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------
st.sidebar.header("🔍 Filters")

# Date filter
date_range = st.sidebar.date_input(
    "Select Date Range",
    [datetime(2025, 1, 1), datetime(2025, 11, 11)]
)

# Dropdown filters
regions = ['North America', 'Europe', 'Asia', 'Africa', 'Latin America']
suppliers = ['Supplier A', 'Supplier B', 'Supplier C', 'Supplier D']
categories = ['Menswear', 'Womenswear', 'Footwear', 'Accessories']

selected_region = st.sidebar.selectbox("Select Region", regions)
selected_supplier = st.sidebar.selectbox("Select Supplier", suppliers)
selected_category = st.sidebar.selectbox("Select Product Category", categories)

# ---------------------------------------
# SIMULATED DATA (Replace with real dataset)
# ---------------------------------------
np.random.seed(42)
data = pd.DataFrame({
    'Date': pd.date_range(start='2025-01-01', periods=200),
    'Lead Time (Days)': np.random.randint(10, 40, size=200),
    'Order Volume': np.random.randint(100, 1200, size=200),
    'Return Rate (%)': np.random.uniform(2, 20, size=200),
    'On-time Delivery (%)': np.random.uniform(70, 100, size=200),
    'Supplier': np.random.choice(suppliers, 200),
    'Region': np.random.choice(regions, 200),
    'Category': np.random.choice(categories, 200)
})

# Filtered Data
filtered_data = data[
    (data['Region'] == selected_region) &
    (data['Supplier'] == selected_supplier) &
    (data['Category'] == selected_category)
]

# ---------------------------------------
# KPI SECTION
# ---------------------------------------
st.subheader("📊 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Avg Lead Time (Days)", f"{filtered_data['Lead Time (Days)'].mean():.1f}")
col2.metric("On-Time Delivery (%)", f"{filtered_data['On-time Delivery (%)'].mean():.1f}")
col3.metric("Return Rate (%)", f"{filtered_data['Return Rate (%)'].mean():.1f}")
col4.metric("Total Orders", f"{filtered_data['Order Volume'].sum():,.0f}")

st.markdown("---")

# ---------------------------------------
# TABS FOR DIFFERENT ANALYSIS
# ---------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Trends & Performance",
    "🚚 Supplier & Logistics",
    "📦 Inventory & Returns",
    "⚠️ Risk & Forecast"
])

# ---------------------------------------
# TAB 1 – Trends & Performance
# ---------------------------------------
with tab1:
    st.subheader("📈 Trend Analysis")

    colA, colB = st.columns(2)

    with colA:
        fig1 = px.line(
            filtered_data, x='Date', y='Lead Time (Days)',
            title="Lead Time Over Time", markers=True
        )
        fig1.update_layout(hovermode='x unified')
        st.plotly_chart(fig1, use_container_width=True)

    with colB:
        fig2 = px.line(
            filtered_data, x='Date', y='Order Volume',
            title="Order Volume Over Time", markers=True
        )
        fig2.update_layout(hovermode='x unified')
        st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------
# TAB 2 – Supplier & Logistics
# ---------------------------------------
with tab2:
    st.subheader("🚚 Supplier Delivery Performance")

    fig3 = px.bar(
        filtered_data, x='Date', y='On-time Delivery (%)',
        color='Supplier', title="On-Time Delivery by Supplier"
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.info("📌 Tip: Identify underperforming suppliers early to avoid delays in seasonal collections.")

# ---------------------------------------
# TAB 3 – Inventory & Returns
# ---------------------------------------
with tab3:
    st.subheader("📦 Inventory & Return Analytics")

    colC, colD = st.columns(2)

    with colC:
        fig4 = px.scatter(
            filtered_data, x='Order Volume', y='Return Rate (%)',
            color='Category', size='Order Volume',
            title="Return Rate vs Order Volume"
        )
        st.plotly_chart(fig4, use_container_width=True)

    with colD:
        st.write("### 🧾 Return Summary by Category")
        summary = filtered_data.groupby("Category")[["Order Volume", "Return Rate (%)"]].mean().reset_index()
        st.dataframe(summary.style.highlight_max(color='lightgreen', axis=0))

# ---------------------------------------
# TAB 4 – Risk & Forecast
# ---------------------------------------
with tab4:
    st.subheader("⚠️ Supplier Risk Assessment")

    pivot = filtered_data.pivot_table(
        index='Region', columns='Supplier',
        values='On-time Delivery (%)', aggfunc='mean'
    )

    st.dataframe(pivot.style.background_gradient(cmap='Reds'))

    st.markdown("### 🔮 Forecast Insight (Simulated Example)")
    st.info("""
    - If current trends continue, Lead Time may increase by 5–7% next quarter.
    - Risk of late deliveries is highest in **Asia** due to supplier capacity constraints.
    - Consider rebalancing sourcing to diversify regional dependencies.
    """)

# ---------------------------------------
# DOWNLOAD SECTION
# ---------------------------------------
st.markdown("---")
st.subheader("📥 Download Filtered Data")

csv = filtered_data.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download CSV Data",
    data=csv,
    file_name="fashion_supply_data.csv",
    mime="text/csv"
)

# ---------------------------------------
# FOOTER
# ---------------------------------------
st.markdown("---")
st.markdown("🧵 **Fashion Supply Chain Dashboard** | Developed using Streamlit + Plotly")
st.caption("© 2025 AnalyticsForFashionSupplyManagement | Enhanced Dashboard Edition")





