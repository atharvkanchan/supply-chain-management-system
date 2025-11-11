import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from datetime import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Fashion Supply Management Analytics", layout="wide")

# ---------------- TITLE & INTRO ----------------
st.title("📊 Analytics for Fashion Supply Management")
st.markdown("""
This dashboard provides insights into fashion supply chain operations — including sales trends, inventory management, supplier performance, 
and demand forecasting with future boom analysis.
Use the sidebar to filter data and explore key metrics.
""")

# ---------------- MOCK DATA (Replace with CSV/DB) ----------------
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

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.header("🔍 Filters")
selected_categories = st.sidebar.multiselect("Select Categories", df["Category"].unique(), default=df["Category"].unique())
selected_products = st.sidebar.multiselect("Select Products", df["Product"].unique(), default=df["Product"].unique())
supplier_filter = st.sidebar.multiselect("Select Suppliers", df["Supplier"].unique(), default=df["Supplier"].unique())
date_range = st.sidebar.date_input("Select Date Range", [df["Date"].min(), df["Date"].max()])

# ---------------- APPLY FILTERS ----------------
filtered_df = df[
    (df["Category"].isin(selected_categories)) &
    (df["Product"].isin(selected_products)) &
    (df["Supplier"].isin(supplier_filter)) &
    (df["Date"] >= pd.to_datetime(date_range[0])) &
    (df["Date"] <= pd.to_datetime(date_range[1]))
]

# ---------------- KPI SECTION ----------------
st.header("📈 Key Performance Indicators (KPIs)")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Sales", f"${filtered_df['Sales'].sum():,.0f}")
with col2:
    st.metric("Avg Inventory", f"{filtered_df['Inventory'].mean():.0f} units")
with col3:
    st.metric("Avg Lead Time", f"{filtered_df['Lead_Time_Days'].mean():.1f} days")
with col4:
    st.metric("Total Cost", f"${filtered_df['Cost'].sum():,.0f}")

# ---------------- VISUALIZATIONS ----------------
st.header("📊 Visualizations")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Sales Trends Over Time")
    sales_trend = filtered_df.groupby("Date")["Sales"].sum().reset_index()
    fig_sales = px.line(sales_trend, x="Date", y="Sales", title="Monthly Sales Trend", markers=True)
    st.plotly_chart(fig_sales, use_container_width=True)

with col2:
    st.subheader("Inventory Levels by Product")
    inventory_bar = px.bar(filtered_df.groupby("Product")["Inventory"].mean().reset_index(),
                           x="Product", y="Inventory", title="Avg Inventory per Product", color="Product")
    st.plotly_chart(inventory_bar, use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    st.subheader("Supplier Performance (Sales by Supplier)")
    supplier_sales = filtered_df.groupby("Supplier")["Sales"].sum().reset_index()
    fig_supplier = px.pie(supplier_sales, names="Supplier", values="Sales", title="Sales Distribution by Supplier")
    st.plotly_chart(fig_supplier, use_container_width=True)

with col4:
    st.subheader("Demand Forecast vs Actual Sales")
    forecast_summary = filtered_df.groupby("Date")[["Sales", "Demand_Forecast"]].sum().reset_index()
    forecast_melted = forecast_summary.melt(id_vars="Date", value_vars=["Sales", "Demand_Forecast"],
                                            var_name="Type", value_name="Value")
    fig_forecast = px.bar(forecast_melted, x="Date", y="Value", color="Type", barmode="group",
                          title="Demand Forecast vs Actual Sales", labels={"Value": "Units", "Type": "Metric"})
    st.plotly_chart(fig_forecast, use_container_width=True)

# ---------------- COST VS SALES ----------------
st.subheader("💰 Cost vs Sales Scatter Plot")
fig_scatter = px.scatter(filtered_df, x="Cost", y="Sales", color="Category", size="Inventory",
                         title="Cost Efficiency Analysis", hover_data=["Product", "Supplier"])
st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------- FORECASTING MODE SWITCH ----------------
st.markdown("---")
st.header("🔮 Future Analytics")

forecast_mode = st.radio(
    "Select Forecast Mode:",
    ["📈 Total Demand Forecast", "🚀 Product Boom Forecast"],
    horizontal=True
)

# ---------------- MODE 1: TOTAL DEMAND FORECAST ----------------
if forecast_mode == "📈 Total Demand Forecast":
    ts = filtered_df.groupby(pd.Grouper(key="Date", freq="M"))["Sales"].sum().reset_index()
    ts = ts.sort_values("Date")

    if len(ts) < 3:
        st.warning("Not enough data for forecasting. Try selecting a wider date range.")
    else:
        X = np.array(ts["Date"].map(datetime.toordinal)).reshape(-1, 1)
        y = np.array(ts["Sales"])
        lr = LinearRegression().fit(X, y)

        horizon = 6
        last_date = ts["Date"].max()
        future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, horizon + 1)]
        future_preds = lr.predict(np.array([d.toordinal() for d in future_dates]).reshape(-1, 1))

        forecast_df = pd.DataFrame({"Date": future_dates, "Predicted_Sales": future_preds})
        combined = pd.concat([
            ts.rename(columns={"Sales": "Value"}).assign(Type="Actual"),
            forecast_df.rename(columns={"Predicted_Sales": "Value"}).assign(Type="Predicted")
        ])

        fig_future = px.line(combined, x="Date", y="Value", color="Type",
                             title="Overall Future Demand Prediction (Next 6 Months)",
                             markers=True)
        st.plotly_chart(fig_future, use_container_width=True)

        next_month_demand = int(future_preds[0])
        st.success(f"📦 Estimated total demand for next month: **{next_month_demand} units (approx.)**")

# ---------------- MODE 2: PRODUCT BOOM FORECAST ----------------
elif forecast_mode == "🚀 Product Boom Forecast":
    st.subheader("🚀 Product-wise Boom Forecast (Next Month Prediction)")

    boom_data = []
    last_date = filtered_df["Date"].max()
    for product, group in filtered_df.groupby("Product"):
        product_ts = group.groupby(pd.Grouper(key="Date", freq="M"))["Sales"].sum().reset_index()
        if len(product_ts) >= 3:
            X_p = np.array(product_ts["Date"].map(datetime.toordinal)).reshape(-1, 1)
            y_p = np.array(product_ts["Sales"])
            model_p = LinearRegression().fit(X_p, y_p)

            next_month = last_date + pd.DateOffset(months=1)
            pred_next = model_p.predict([[next_month.toordinal()]])[0]
            last_sales = product_ts.iloc[-1]["Sales"]
            growth = ((pred_next - last_sales) / last_sales) * 100 if last_sales > 0 else 0
            boom_data.append({"Product": product, "Predicted_Sales": pred_next, "Growth_%": growth})

    boom_df = pd.DataFrame(boom_data).sort_values("Predicted_Sales", ascending=False)

    if not boom_df.empty:
        fig_boom = px.bar(
            boom_df, x="Product", y="Predicted_Sales", color="Growth_%",
            text=boom_df["Growth_%"].apply(lambda x: f"{x:.1f}%"),
            title="Top Products Expected to Boom Next Month",
            color_continuous_scale="Tealgrn"
        )
        fig_boom.update_traces(textposition="outside")
        st.plotly_chart(fig_boom, use_container_width=True)

        top_boom = boom_df.iloc[0]
        st.success(
            f"🔥 **{top_boom['Product']}** is projected to be next month's boom product "
            f"with estimated **{int(top_boom['Predicted_Sales'])} units** (+{top_boom['Growth_%']:.1f}% growth)."
        )
    else:
        st.info("Not enough data to determine booming products.")

# ---------------- DATA TABLE & DOWNLOAD ----------------
st.header("📋 Filtered Data")
st.dataframe(filtered_df)

csv = filtered_df.to_csv(index=False)
st.download_button("Download Filtered Data as CSV", csv, "fashion_supply_data.csv", "text/csv")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("""
🧵 Dashboard built with **Streamlit**  
💡 Replace mock data with your real CSV or database for live analytics.  
📈 Features: Sales trends, supplier insights, inventory tracking, and future demand forecasting.
""")
