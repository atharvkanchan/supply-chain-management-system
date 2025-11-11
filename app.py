import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Set page config
st.set_page_config(page_title="Fashion Supply Management Analytics", layout="wide")

# Title and description
st.title("📊 Analytics for Fashion Supply Management")
st.markdown("""
This dashboard provides insights into fashion supply chain operations, including sales trends, inventory management, supplier performance, and demand forecasting.
Use the sidebar to filter data and explore key metrics.
""")

# Generate mock data (replace with your real data source, e.g., CSV, database, or API)
np.random.seed(42)
dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="M")
products = ["T-Shirts", "Jeans", "Dresses", "Shoes", "Accessories"]
categories = ["Men", "Women", "Unisex"]
suppliers = ["Supplier A", "Supplier B", "Supplier C", "Supplier D"]

data = []
for _ in range(500):
    data.append({
        "Date": np.random.choice(dates),
        "Product": np.random.choice(products),
        "Category": np.random.choice(categories),
        "Supplier": np.random.choice(suppliers),
        "Sales": np.random.randint(100, 1000),
        "Inventory": np.random.randint(50, 500),
        "Demand_Forecast": np.random.randint(80, 1200),
        "Lead_Time_Days": np.random.randint(5, 30),
        "Cost": np.random.uniform(10, 200)
    })

df = pd.DataFrame(data)
df["Date"] = pd.to_datetime(df["Date"])

# Sidebar filters
st.sidebar.header("🔍 Filters")
selected_categories = st.sidebar.multiselect("Select Categories", options=df["Category"].unique(), default=df["Category"].unique())
selected_products = st.sidebar.multiselect("Select Products", options=df["Product"].unique(), default=df["Product"].unique())
date_range = st.sidebar.date_input("Select Date Range", [df["Date"].min(), df["Date"].max()])
supplier_filter = st.sidebar.multiselect("Select Suppliers", options=df["Supplier"].unique(), default=df["Supplier"].unique())

# Apply filters
filtered_df = df[
    (df["Category"].isin(selected_categories)) &
    (df["Product"].isin(selected_products)) &
    (df["Supplier"].isin(supplier_filter)) &
    (df["Date"] >= pd.to_datetime(date_range[0])) &
    (df["Date"] <= pd.to_datetime(date_range[1]))
]

# KPIs
st.header("Key Performance Indicators (KPIs)")
col1, col2, col3, col4 = st.columns(4)
with col1:
    total_sales = filtered_df["Sales"].sum()
    st.metric("Total Sales", f"${total_sales:,.0f}")
with col2:
    avg_inventory = filtered_df["Inventory"].mean()
    st.metric("Avg Inventory", f"{avg_inventory:.0f} units")
with col3:
    avg_lead_time = filtered_df["Lead_Time_Days"].mean()
    st.metric("Avg Lead Time", f"{avg_lead_time:.1f} days")
with col4:
    total_cost = filtered_df["Cost"].sum()
    st.metric("Total Cost", f"${total_cost:,.0f}")

# Charts section
st.header("📈 Visualizations")

# Row 1: Sales Trends and Inventory Levels
col1, col2 = st.columns(2)
with col1:
    st.subheader("Sales Trends Over Time")
    sales_trend = filtered_df.groupby("Date")["Sales"].sum().reset_index()
    fig_sales = px.line(sales_trend, x="Date", y="Sales", title="Monthly Sales", markers=True)
    st.plotly_chart(fig_sales, use_container_width=True)

with col2:
    st.subheader("Inventory Levels by Product")
    inventory_bar = px.bar(filtered_df.groupby("Product")["Inventory"].mean().reset_index(), 
                           x="Product", y="Inventory", title="Avg Inventory per Product", color="Product")
    st.plotly_chart(inventory_bar, use_container_width=True)

# Row 2: Supplier Performance and Demand Forecast
col3, col4 = st.columns(2)
with col3:
    st.subheader("Supplier Performance (Sales by Supplier)")
    supplier_sales = filtered_df.groupby("Supplier")["Sales"].sum().reset_index()
    fig_supplier = px.pie(supplier_sales, names="Supplier", values="Sales", title="Sales Distribution by Supplier")
    st.plotly_chart(fig_supplier, use_container_width=True)

with col4:
    st.subheader("Demand Forecast vs Actual Sales")
    forecast_vs_actual = filtered_df.melt(id_vars=["Date"], value_vars=["Sales", "Demand_Forecast"], 
                                          var_name="Type", value_name="Value")
    fig_forecast = px.line(forecast_vs_actual, x="Date", y="Value", color="Type", 
                           title="Demand Forecast vs Actual Sales", markers=True)
    st.plotly_chart(fig_forecast, use_container_width=True)

# Additional: Scatter plot for Cost vs Sales
st.subheader("Cost vs Sales Scatter Plot")
fig_scatter = px.scatter(filtered_df, x="Cost", y="Sales", color="Category", size="Inventory", 
                         title="Cost Efficiency Analysis", hover_data=["Product", "Supplier"])
st.plotly_chart(fig_scatter, use_container_width=True)

# Data Table
st.header("📋 Filtered Data")
st.dataframe(filtered_df)

# Download option
csv = filtered_df.to_csv(index=False)
st.download_button(label="Download Filtered Data as CSV", data=csv, file_name="fashion_supply_data.csv", mime="text/csv")

# Footer
st.markdown("---")
st.markdown("Dashboard built with Streamlit. Mock data used for demonstration – replace with real data (e.g., from ERP systems like SAP or databases). For customizations or integrations (e.g., ML forecasting), provide more details!")
