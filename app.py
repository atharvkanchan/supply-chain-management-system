# app.py
# Streamlit Fashion / Supply Chain Dashboard (uses converted CSV)
# Place this file in the same folder as 'india_supply_chain_converted.csv'

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from datetime import datetime

st.set_page_config(page_title="Fashion Supply Chain Management Dashboard", layout="wide")

st.title("📊 Fashion Supply Chain Management Analytics")
st.markdown("""
This dashboard loads the converted dataset `india_supply_chain_converted.csv` (synthetic columns added)
and presents KPIs, visualizations, forecasting and inventory helpers.
""")

# Primary CSV path (local folder)
CSV_PATH = "india_supply_chain_converted.csv"

# Allow user to upload file if not found
uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV)")

@st.cache_data
def load_data(path=CSV_PATH):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"Could not find {path}. Make sure the CSV is in the same folder as app.py")
        st.stop()

    # Standardize column names (strip + lower) but preserve originals for convenience
    df.columns = [c.strip() for c in df.columns]

    # Ensure Date column exists or try to infer
    if "Date" not in df.columns:
        # Attempt to find a date-like column
        for c in df.columns:
            if "date" in c.lower() or "timestamp" in c.lower():
                df.rename(columns={c: "Date"}, inplace=True)
                break

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
    else:
        # If no date, create a synthetic monthly index to allow plotting
        df["Date"] = pd.date_range(start="2023-01-01", periods=len(df), freq="D")

    # Ensure required columns exist (they were added to the converted CSV)
    required = ["Product", "Category", "Sales", "Inventory", "Lead_Time_Days", "Cost"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}. Use the converted CSV that contains these columns.")
        st.stop()

    # Coerce numeric columns
    for num_col in ["Sales", "Inventory", "Lead_Time_Days", "Cost"]:
        df[num_col] = pd.to_numeric(df[num_col], errors="coerce").fillna(0)

    return df

# Load data
# Load data with fallback to upload
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = load_data()()

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.header("🔍 Filters")

all_categories = sorted(df["Category"].dropna().unique().tolist())
all_products = sorted(df["Product"].dropna().unique().tolist())
all_suppliers = sorted(df["Supplier"].dropna().unique().tolist()) if "Supplier" in df.columns else []

selected_categories = st.sidebar.multiselect("Select Categories", all_categories, default=all_categories)
selected_products = st.sidebar.multiselect("Select Products", all_products, default=all_products)

if all_suppliers:
    selected_suppliers = st.sidebar.multiselect("Select Suppliers", all_suppliers, default=all_suppliers)
else:
    selected_suppliers = None

# Date range
min_date = df["Date"].min().date()
max_date = df["Date"].max().date()
start_date, end_date = st.sidebar.date_input("Select Date Range (start, end)", [min_date, max_date])
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Apply filters
filtered = df[
    (df["Category"].isin(selected_categories)) &
    (df["Product"].isin(selected_products)) &
    (df["Date"] >= start_date) &
    (df["Date"] <= end_date)
].copy()

if selected_suppliers is not None:
    filtered = filtered[filtered["Supplier"].isin(selected_suppliers)] if "Supplier" in filtered.columns else filtered

# Early exit for empty
if filtered.empty:
    st.warning("No data matches your filters. Adjust filters or date range to see charts and KPIs.")
    st.dataframe(filtered)
    st.stop()

# ---------------- KPI SECTION ----------------
st.header("📈 Key Performance Indicators (KPIs)")
col1, col2, col3, col4 = st.columns(4)

total_sales = int(filtered['Sales'].sum())
avg_inventory = filtered['Inventory'].mean()
avg_lead = filtered['Lead_Time_Days'].mean()
total_cost = float(filtered['Cost'].sum())

with col1:
    st.metric("Total Sales", f"{total_sales:,}")
with col2:
    st.metric("Average Inventory", f"{avg_inventory:.0f} units")
with col3:
    st.metric("Average Lead Time", f"{avg_lead:.1f} days")
with col4:
    st.metric("Total Cost", f"₹{total_cost:,.0f}")

# ---------------- VISUAL ANALYTICS ----------------
st.header("📊 Visual Analytics")

# Sales trend
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Sales Trends Over Time")
    sales_trend = filtered.groupby(pd.Grouper(key="Date", freq="M"))["Sales"].sum().reset_index().sort_values("Date")
    fig_sales = px.line(sales_trend, x="Date", y="Sales", title="Monthly Sales Trend", markers=True)
    st.plotly_chart(fig_sales, use_container_width=True)

with col_b:
    st.subheader("Inventory Levels by Product")
    inv = filtered.groupby("Product")["Inventory"].mean().reset_index()
    fig_inv = px.bar(inv, x="Product", y="Inventory", title="Average Inventory per Product", color="Product")
    st.plotly_chart(fig_inv, use_container_width=True)

# Supplier and forecast
col_c, col_d = st.columns(2)
with col_c:
    st.subheader("Supplier Performance")
    if "Supplier" in filtered.columns:
        sup_sales = filtered.groupby("Supplier")["Sales"].sum().reset_index()
        fig_sup = px.pie(sup_sales, names="Supplier", values="Sales", title="Sales Distribution by Supplier", hole=0.4)
        st.plotly_chart(fig_sup, use_container_width=True)
    else:
        st.info("No 'Supplier' column in dataset.")

with col_d:
    st.subheader("Demand Forecast vs Actual Sales")
    fc = filtered.groupby(pd.Grouper(key="Date", freq="M"))["Sales"].sum().reset_index().sort_values("Date")
    if not fc.empty:
        fc["Demand_Forecast"] = fc["Sales"].rolling(2, min_periods=1).mean() * np.random.uniform(0.9, 1.1)
        fm = fc.melt(id_vars="Date", value_vars=["Sales", "Demand_Forecast"], var_name="Type", value_name="Value")
        fig_fc = px.bar(fm, x="Date", y="Value", color="Type", barmode="group", title="Demand Forecast vs Actual Sales")
        st.plotly_chart(fig_fc, use_container_width=True)

# Cost vs Sales
st.subheader("💰 Cost vs Sales Scatter Plot")
fig_scatter = px.scatter(filtered, x="Cost", y="Sales", color="Category", size="Inventory", title="Cost Efficiency Analysis", hover_data=["Product"])
st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------- FUTURE ANALYTICS ----------------
st.markdown("---")
st.header("🔮 Future Analytics")
forecast_mode = st.radio("Select Forecast Mode:", ["📈 Total Demand Forecast", "🚀 Product Boom Forecast"], horizontal=True)

# Mode 1
if forecast_mode == "📈 Total Demand Forecast":
    ts = filtered.groupby(pd.Grouper(key="Date", freq="M"))["Sales"].sum().reset_index().sort_values("Date")
    if len(ts) >= 3:
        X = np.array(ts["Date"].map(lambda d: d.to_pydatetime().toordinal())).reshape(-1, 1)
        y = np.array(ts["Sales"])
        lr = LinearRegression().fit(X, y)
        future_dates = [ts["Date"].max() + pd.DateOffset(months=i) for i in range(1, 7)]
        preds = lr.predict(np.array([d.to_pydatetime().toordinal() for d in future_dates]).reshape(-1, 1))

        forecast_df = pd.DataFrame({"Date": future_dates, "Predicted_Sales": preds})
        combined = pd.concat([
            ts.rename(columns={"Sales": "Value"}).assign(Type="Actual"),
            forecast_df.rename(columns={"Predicted_Sales": "Value"}).assign(Type="Predicted")
        ])
        fig_future = px.line(combined, x="Date", y="Value", color="Type", title="6-Month Future Demand Forecast", markers=True)
        st.plotly_chart(fig_future, use_container_width=True)
        st.success(f"📦 Next Month Estimated Demand: **{int(preds[0])} units (approx.)**")
    else:
        st.warning("Not enough data for forecasting (need at least 3 monthly points).")

# Mode 2
else:
    st.subheader("🚀 Product-wise Boom Forecast (Next Month Prediction)")
    boom_data = []
    last_date = filtered["Date"].max()
    for product, group in filtered.groupby("Product"):
        product_ts = group.groupby(pd.Grouper(key="Date", freq="M"))["Sales"].sum().reset_index().sort_values("Date")
        if len(product_ts) >= 3:
            X_p = np.array(product_ts["Date"].map(lambda d: d.to_pydatetime().toordinal())).reshape(-1, 1)
            y_p = np.array(product_ts["Sales"])
            model_p = LinearRegression().fit(X_p, y_p)
            next_month = last_date + pd.DateOffset(months=1)
            pred_next = model_p.predict(np.array([[next_month.to_pydatetime().toordinal()]]))[0]
            last_sales = product_ts.iloc[-1]["Sales"]
            growth = ((pred_next - last_sales) / last_sales) * 100 if last_sales > 0 else 0
            boom_data.append({"Product": product, "Predicted_Sales": pred_next, "Growth_%": growth})

    boom_df = pd.DataFrame(boom_data).sort_values("Predicted_Sales", ascending=False)
    if not boom_df.empty:
        fig_boom = px.bar(boom_df, x="Product", y="Predicted_Sales", color="Growth_%", text=boom_df["Growth_%"].apply(lambda x: f"{x:.1f}%"), title="Top Products Expected to Boom Next Month")
        fig_boom.update_traces(textposition="outside")
        st.plotly_chart(fig_boom, use_container_width=True)
        top_boom = boom_df.iloc[0]
        st.success(f"🔥 {top_boom['Product']} projected to boom: **{int(top_boom['Predicted_Sales'])} units** (+{top_boom['Growth_%']:.1f}% growth).")
    else:
        st.info("Not enough data for booming products.")

# ---------------- INVENTORY OPTIMIZATION ----------------
st.markdown("---")
st.header("📦 Smart Inventory Optimization & Reorder Alerts")

inventory_df = filtered.groupby("Product").agg({
    "Sales": "mean", "Inventory": "mean", "Lead_Time_Days": "mean"
}).reset_index()

inventory_df["Reorder_Level"] = (inventory_df["Sales"] * (inventory_df["Lead_Time_Days"] / 7)).round()
inventory_df["Status"] = np.where(inventory_df["Inventory"] < inventory_df["Reorder_Level"], "⚠️ Low Stock", "✅ Sufficient")

fig_inv = px.bar(inventory_df, x="Product", y=["Inventory", "Reorder_Level"], barmode="group", title="Inventory vs Reorder Threshold")
st.plotly_chart(fig_inv, use_container_width=True)

low_stock = inventory_df[inventory_df["Status"] == "⚠️ Low Stock"].copy()
if not low_stock.empty:
    st.warning("⚠️ The following products are below safe stock levels:")
    low_stock["Suggested_Reorder_Qty"] = (low_stock["Reorder_Level"] * 1.5 - low_stock["Inventory"]).clip(lower=0).astype(int)
    st.dataframe(low_stock[["Product", "Inventory", "Reorder_Level", "Suggested_Reorder_Qty", "Status"]])
else:
    st.success("✅ All products are above their safe stock levels.")

# ---------------- DATA TABLE & DOWNLOAD ----------------
st.header("📋 Filtered Dataset View")
st.dataframe(filtered)

csv = filtered.to_csv(index=False)
st.download_button("⬇️ Download Filtered Data as CSV", csv, "fashion_supply_data_filtered.csv", "text/csv")

st.markdown("---")
st.markdown("🧵 **Fashion Supply Management Dashboard** — Built with ❤️ using Streamlit and Plotly")
