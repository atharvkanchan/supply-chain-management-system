# =============================================================
# 📦 India Supply Chain Management Dashboard (For Your Dataset)
# =============================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from datetime import datetime

st.set_page_config(page_title="Supply Chain Management Dashboard", layout="wide")

st.title("📊 India Supply Chain Management Analytics")
st.markdown("""
Welcome to the **Supply Chain Management Dashboard**.  
This dashboard provides insights into **inventory, sales, suppliers, lead time, and forecasting**.
""")

# -------------------- LOAD YOUR DATASET --------------------
@st.cache_data
def load_data():
    df = pd.read_csv("india_supply_chain_2024_2025.csv")
    # Try to detect date column names automatically
    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df.rename(columns={col: "Date"}, inplace=True)
            break
    # If no date column = error
    if "Date" not in df.columns:
        st.error("❌ No date column found in dataset. Add a column named 'Date'.")
        st.stop()
    return df

df = load_data()

# -------------------- STANDARDIZE EXPECTED COLUMNS --------------------
expected_cols = {
    "Product": ["product", "item", "sku"],
    "Category": ["category", "type", "segment"],
    "Supplier": ["supplier", "vendor"],
    "Sales": ["sales", "quantity_sold", "units_sold"],
    "Inventory": ["inventory", "stock", "qty"],
    "Lead_Time_Days": ["lead_time", "lead_days", "delivery_time"],
    "Cost": ["cost", "unit_cost", "price"]
}

# Auto-match columns
for key, options in expected_cols.items():
    if key not in df.columns:
        for col in df.columns:
            if col.lower() in options:
                df.rename(columns={col: key}, inplace=True)
                break

# Validate all required columns
required = ["Date", "Product", "Category", "Supplier", "Sales", "Inventory", "Lead_Time_Days", "Cost"]
missing = [col for col in required if col not in df.columns]
if missing:
    st.error(f"❌ Missing required columns in your dataset: {missing}")
    st.stop()

# Clean
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df.dropna(subset=["Date"], inplace=True)

# -------------------- SIDEBAR FILTERS --------------------
st.sidebar.header("🔍 Filters")

selected_categories = st.sidebar.multiselect(
    "Select Categories", df["Category"].unique(), default=list(df["Category"].unique())
)

selected_products = st.sidebar.multiselect(
    "Select Products", df["Product"].unique(), default=list(df["Product"].unique())
)

selected_suppliers = st.sidebar.multiselect(
    "Select Suppliers", df["Supplier"].unique(), default=list(df["Supplier"].unique())
)

date_range = st.sidebar.date_input(
    "Select Date Range",
    [df["Date"].min().date(), df["Date"].max().date()]
)

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range

start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Apply filters
filtered_df = df[
    (df["Category"].isin(selected_categories)) &
    (df["Product"].isin(selected_products)) &
    (df["Supplier"].isin(selected_suppliers)) &
    (df["Date"] >= start_date) &
    (df["Date"] <= end_date)
].copy()

# Empty data handle
if filtered_df.empty:
    st.warning("⚠️ No data available for selected filters.")
    st.dataframe(filtered_df)
    st.stop()

# -------------------- KPIs --------------------
st.header("📈 Key Performance Indicators (KPIs)")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Sales", f"{int(filtered_df['Sales'].sum()):,}")

with col2:
    st.metric("Average Inventory", f"{filtered_df['Inventory'].mean():.0f}")

with col3:
    st.metric("Average Lead Time", f"{filtered_df['Lead_Time_Days'].mean():.1f} days")

with col4:
    st.metric("Total Cost", f"₹{filtered_df['Cost'].sum():,.0f}")

# -------------------- VISUAL ANALYTICS --------------------
st.header("📊 Visual Analytics")

col1, col2 = st.columns(2)

# Sales Trend
with col1:
    st.subheader("Sales Trends Over Time")
    sales_trend = filtered_df.groupby("Date")["Sales"].sum().reset_index()
    fig_sales = px.line(sales_trend, x="Date", y="Sales", markers=True)
    st.plotly_chart(fig_sales, use_container_width=True)

# Inventory by Product
with col2:
    st.subheader("Inventory Levels by Product")
    inv = filtered_df.groupby("Product")["Inventory"].mean().reset_index()
    fig_inv = px.bar(inv, x="Product", y="Inventory", color="Product")
    st.plotly_chart(fig_inv, use_container_width=True)

# Supplier Performance
col3, col4 = st.columns(2)

with col3:
    st.subheader("Supplier Performance")
    sup = filtered_df.groupby("Supplier")["Sales"].sum().reset_index()
    fig_sup = px.pie(sup, names="Supplier", values="Sales", hole=0.45)
    st.plotly_chart(fig_sup, use_container_width=True)

# Forecast vs Actual
with col4:
    st.subheader("Demand Forecast vs Sales")
    fc = filtered_df.groupby("Date")["Sales"].sum().reset_index()
    fc["Demand_Forecast"] = fc["Sales"].rolling(2, min_periods=1).mean() * np.random.uniform(0.9, 1.1)
    melted = fc.melt(id_vars="Date", var_name="Type", value_name="Value")
    fig_fc = px.bar(melted, x="Date", y="Value", color="Type", barmode="group")
    st.plotly_chart(fig_fc, use_container_width=True)

# Cost vs Sales Scatter
st.subheader("💰 Cost vs Sales")
fig_scatter = px.scatter(
    filtered_df, x="Cost", y="Sales", color="Product",
    size="Inventory", hover_data=["Supplier"]
)
st.plotly_chart(fig_scatter, use_container_width=True)

# -------------------- DEMAND FORECAST --------------------
st.header("🔮 Future Forecasting")

mode = st.radio("Choose Forecast Mode", ["Total Demand Forecast", "Product Boom Forecast"], horizontal=True)

if mode == "Total Demand Forecast":
    ts = filtered_df.groupby(pd.Grouper(key="Date", freq="M"))["Sales"].sum().reset_index()

    if len(ts) >= 3:
        X = ts["Date"].map(lambda x: x.toordinal()).values.reshape(-1, 1)
        y = ts["Sales"].values

        model = LinearRegression().fit(X, y)

        future_dates = [ts["Date"].max() + pd.DateOffset(months=i) for i in range(1, 7)]
        preds = model.predict(np.array([d.toordinal() for d in future_dates]).reshape(-1, 1))

        forecast_df = pd.DataFrame({"Date": future_dates, "Predicted_Sales": preds})

        combined = pd.concat([
            ts.rename(columns={"Sales": "Value"}).assign(Type="Actual"),
            forecast_df.rename(columns={"Predicted_Sales": "Value"}).assign(Type="Predicted")
        ])

        fig_future = px.line(combined, x="Date", y="Value", color="Type", markers=True)
        st.plotly_chart(fig_future, use_container_width=True)

        st.success(f"📦 Next Month Demand Estimate: **{int(preds[0])} units**")

    else:
        st.warning("Not enough data for demand forecast (need minimum 3 months).")

else:
    st.subheader("📈 Next Month Boom Product Prediction")
    results = []
    last_date = filtered_df["Date"].max()

    for product, group in filtered_df.groupby("Product"):
        monthly = group.groupby(pd.Grouper(key="Date", freq="M"))["Sales"].sum().reset_index()
        if len(monthly) >= 3:
            X = monthly["Date"].map(lambda x: x.toordinal()).values.reshape(-1, 1)
            y = monthly["Sales"].values
            lr = LinearRegression().fit(X, y)

            next_month = last_date + pd.DateOffset(months=1)
            pred = lr.predict([[next_month.toordinal()]])[0]
            last_sales = monthly.iloc[-1]["Sales"]
            growth = ((pred - last_sales) / last_sales) * 100 if last_sales > 0 else 0

            results.append({"Product": product, "Predicted_Sales": pred, "Growth_%": growth})

    boom_df = pd.DataFrame(results).sort_values("Predicted_Sales", ascending=False)

    fig_boom = px.bar(
        boom_df, x="Product", y="Predicted_Sales",
        color="Growth_%", text=boom_df["Growth_%"].apply(lambda x: f"{x:.1f}%")
    )
    st.plotly_chart(fig_boom, use_container_width=True)

# -------------------- INVENTORY OPTIMIZATION --------------------
st.header("📦 Inventory Optimization")

inv_df = filtered_df.groupby("Product").agg({
    "Sales": "mean",
    "Inventory": "mean",
    "Lead_Time_Days": "mean"
}).reset_index()

inv_df["Reorder_Level"] = (inv_df["Sales"] * (inv_df["Lead_Time_Days"] / 7)).round()
inv_df["Status"] = np.where(inv_df["Inventory"] < inv_df["Reorder_Level"], "⚠️ Low", "✅ OK")

fig_reorder = px.bar(inv_df, x="Product", y=["Inventory", "Reorder_Level"], barmode="group")
st.plotly_chart(fig_reorder, use_container_width=True)

low = inv_df[inv_df["Status"] == "⚠️ Low"]

if not low.empty:
    low["Reorder_Qty"] = (low["Reorder_Level"] * 1.5 - low["Inventory"]).clip(lower=0).astype(int)
    st.warning("⚠️ Low Stock Items")
    st.dataframe(low)
else:
    st.success("✅ All inventory levels are safe.")

# -------------------- DOWNLOAD --------------------
st.header("📥 Data Preview & Download")
st.dataframe(filtered_df)

csv = filtered_df.to_csv(index=False)
st.download_button("⬇️ Download CSV", csv, "filtered_supply_chain.csv", "text/csv")

st.markdown("----")
st.markdown("🧵 **Supply Chain Dashboard — Built with ❤️ using Streamlit and Plotly**")
