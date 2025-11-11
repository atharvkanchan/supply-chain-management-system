import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Optional imports for advanced forecasting
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False

# Page config
st.set_page_config(page_title="Fashion Supply Management Analytics", layout="wide")

# Title & description
st.title("📊 Analytics for Fashion Supply Management")
st.markdown("""
This dashboard provides insights into fashion supply chain operations — including sales trends, inventory management, supplier performance, 
demand forecasting, and future demand prediction. Use the sidebar to filter data and explore key metrics.
""")

# ---------- Mock Data (replace with real CSV / DB) ----------
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

# ---------- Sidebar filters & Forecast options ----------
st.sidebar.header("🔍 Filters")
selected_categories = st.sidebar.multiselect("Select Categories", options=df["Category"].unique(), default=df["Category"].unique())
selected_products = st.sidebar.multiselect("Select Products", options=df["Product"].unique(), default=df["Product"].unique())
date_range = st.sidebar.date_input("Select Date Range", [df["Date"].min(), df["Date"].max()])
supplier_filter = st.sidebar.multiselect("Select Suppliers", options=df["Supplier"].unique(), default=df["Supplier"].unique())

st.sidebar.markdown("---")
st.sidebar.header("🔮 Forecast Settings")
model_choice = st.sidebar.selectbox("Choose Forecast Model", options=["Linear Regression", "Prophet (if installed)", "SARIMAX (ARIMA)"])
horizon_months = st.sidebar.slider("Forecast Horizon (months)", min_value=1, max_value=12, value=6)

if model_choice == "Prophet (if installed)" and not PROPHET_AVAILABLE:
    st.sidebar.error("Prophet not installed. Run: pip install prophet")

if model_choice == "SARIMAX (ARIMA)" and not STATSMODELS_AVAILABLE:
    st.sidebar.error("statsmodels not installed. Run: pip install statsmodels")

# ---------- Apply filters ----------
filtered_df = df[
    (df["Category"].isin(selected_categories)) &
    (df["Product"].isin(selected_products)) &
    (df["Supplier"].isin(supplier_filter)) &
    (df["Date"] >= pd.to_datetime(date_range[0])) &
    (df["Date"] <= pd.to_datetime(date_range[1]))
].copy()

# ---------- KPIs ----------
st.header("📈 Key Performance Indicators (KPIs)")
col1, col2, col3, col4 = st.columns(4)
with col1:
    total_sales = filtered_df["Sales"].sum()
    st.metric("Total Sales", f"${total_sales:,.0f}")
with col2:
    avg_inventory = filtered_df["Inventory"].mean() if not filtered_df.empty else 0
    st.metric("Avg Inventory", f"{avg_inventory:.0f} units")
with col3:
    avg_lead_time = filtered_df["Lead_Time_Days"].mean() if not filtered_df.empty else 0
    st.metric("Avg Lead Time", f"{avg_lead_time:.1f} days")
with col4:
    total_cost = filtered_df["Cost"].sum()
    st.metric("Total Cost", f"${total_cost:,.0f}")

# ---------- Visualizations (unchanged) ----------
st.header("📊 Visualizations")
# Sales Trends
col1, col2 = st.columns(2)
with col1:
    st.subheader("Sales Trends Over Time")
    sales_trend = filtered_df.groupby("Date")["Sales"].sum().reset_index()
    if not sales_trend.empty:
        fig_sales = px.line(sales_trend, x="Date", y="Sales", title="Monthly Sales Trend", markers=True)
        st.plotly_chart(fig_sales, use_container_width=True)
    else:
        st.info("No data for selected filters to show Sales Trend.")

with col2:
    st.subheader("Inventory Levels by Product")
    if not filtered_df.empty:
        inventory_bar = px.bar(filtered_df.groupby("Product")["Inventory"].mean().reset_index(),
                               x="Product", y="Inventory", title="Avg Inventory per Product", color="Product")
        st.plotly_chart(inventory_bar, use_container_width=True)
    else:
        st.info("No data for selected filters to show Inventory Levels.")

# Supplier & Forecast Comparison
col3, col4 = st.columns(2)
with col3:
    st.subheader("Supplier Performance (Sales by Supplier)")
    if not filtered_df.empty:
        supplier_sales = filtered_df.groupby("Supplier")["Sales"].sum().reset_index()
        fig_supplier = px.pie(supplier_sales, names="Supplier", values="Sales", title="Sales Distribution by Supplier")
        st.plotly_chart(fig_supplier, use_container_width=True)
    else:
        st.info("No data for selected filters to show Supplier Performance.")

with col4:
    st.subheader("Demand Forecast vs Actual Sales")
    if not filtered_df.empty:
        forecast_summary = filtered_df.groupby("Date")[["Sales", "Demand_Forecast"]].sum().reset_index()
        forecast_melted = forecast_summary.melt(id_vars="Date", value_vars=["Sales", "Demand_Forecast"],
                                                var_name="Type", value_name="Value")
        fig_forecast = px.bar(forecast_melted, x="Date", y="Value", color="Type", barmode="group",
                              title="Demand Forecast vs Actual Sales", labels={"Value": "Units", "Type": "Metric"})
        st.plotly_chart(fig_forecast, use_container_width=True)
    else:
        st.info("No data for selected filters to show Forecast vs Actual.")

# Cost vs Sales scatter
st.subheader("💰 Cost vs Sales Scatter Plot")
if not filtered_df.empty:
    fig_scatter = px.scatter(filtered_df, x="Cost", y="Sales", color="Category", size="Inventory",
                             title="Cost Efficiency Analysis", hover_data=["Product", "Supplier"])
    st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.info("No data to show Cost vs Sales scatter.")

# ---------- Forecasting Section (LinearRegression / Prophet / SARIMAX) ----------
st.header("🔮 Future Demand Prediction (Model Comparison)")

# Prepare monthly aggregated series
if filtered_df.empty:
    st.warning("No data available for forecasting with current filters.")
else:
    ts = filtered_df.groupby(pd.Grouper(key="Date", freq="M"))["Sales"].sum().reset_index()
    ts = ts.sort_values("Date").reset_index(drop=True)

    # require at least 6 data points for reasonable forecasting
    if len(ts) < 6:
        st.warning("Need at least 6 months of data to run reliable forecasts. Try expanding the date range.")
    else:
        # Train/test split: last 3 months as test
        test_months = min(3, max(1, len(ts)//6))  # small holdout
        train_ts = ts.iloc[:-test_months].copy()
        test_ts = ts.iloc[-test_months:].copy()

        # Sidebar: show chosen product(s) summary
        st.markdown(f"**Model:** {model_choice} — **Horizon:** {horizon_months} months")
        st.markdown(f"Training on {len(train_ts)} months, testing on {len(test_ts)} months.")

        # Helper: create future dates
        last_train_date = train_ts["Date"].max()
        future_dates = [last_train_date + relativedelta(months=i) for i in range(1, horizon_months + 1)]

        # ---------- Linear Regression ----------
        if model_choice == "Linear Regression":
            # convert date to ordinal
            X_train = np.array(train_ts["Date"].map(datetime.toordinal)).reshape(-1, 1)
            y_train = np.array(train_ts["Sales"])
            X_test = np.array(test_ts["Date"].map(datetime.toordinal)).reshape(-1, 1)
            y_test = np.array(test_ts["Sales"])

            lr = LinearRegression()
            lr.fit(X_train, y_train)

            # Historical in-sample predictions (for plotting)
            train_preds = lr.predict(np.array(ts["Date"].map(datetime.toordinal)).reshape(-1, 1))

            # Future predictions
            future_ordinal = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
            future_preds = lr.predict(future_ordinal)

            # Evaluate on test set
            test_preds = lr.predict(X_test)
            mae = mean_absolute_error(y_test, test_preds)

            # Compose plotting dataframe
            plot_df = pd.DataFrame({
                "Date": pd.concat([ts["Date"], pd.Series(future_dates)]),
                "Sales": np.concatenate([ts["Sales"].values, np.repeat(np.nan, len(future_dates))]),
                "Predicted": np.concatenate([train_preds, future_preds])
            })

            # Plot actual vs predicted (actual historical shown, predicted for historical+future)
            fig = px.line(plot_df, x="Date", y=["Sales", "Predicted"],
                          labels={"value": "Sales", "variable": "Series"},
                          title=f"Linear Regression Forecast (MAE on holdout: {mae:.2f})")
            st.plotly_chart(fig, use_container_width=True)

            st.success(f"Estimated demand next month (LinearRegression): {int(future_preds[0])} units (approx.)")

        # ---------- Prophet ----------
        elif model_choice == "Prophet (if installed)":
            if not PROPHET_AVAILABLE:
                st.error("Prophet is not installed in this environment. Install with `pip install prophet` and restart the app.")
            else:
                # Prophet expects columns ds and y
                prophet_df = train_ts.rename(columns={"Date": "ds", "Sales": "y"})[["ds", "y"]]
                m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                m.fit(prophet_df)

                # Create future dataframe and forecast
                future = m.make_future_dataframe(periods=horizon_months, freq='M')
                forecast = m.predict(future)

                # Split forecast into historical fitted and future
                forecast_plot = forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Predicted'})

                # Evaluate MAE on test (use model to predict test dates)
                test_prophet = forecast_plot[forecast_plot['Date'].isin(test_ts['Date'])]
                if not test_prophet.empty:
                    mae = mean_absolute_error(test_ts['Sales'].values, test_prophet['Predicted'].values)
                else:
                    mae = float('nan')

                # Plot
                actual_and_pred = pd.concat([
                    ts.rename(columns={"Sales": "Sales"}),
                    forecast_plot[forecast_plot['Date'] > ts['Date'].max()][['Date', 'Predicted']].assign(Sales=np.nan)
                ], ignore_index=True, sort=False)

                fig = px.line(actual_and_pred, x="Date", y=["Sales", "Predicted"],
                              labels={"value": "Sales", "variable": "Series"},
                              title=f"Prophet Forecast (MAE on holdout: {mae:.2f})")
                st.plotly_chart(fig, use_container_width=True)

                # next month estimate from forecast
                next_month_pred = int(forecast_plot[forecast_plot['Date'] == future_dates[0]]['Predicted'].values[0])
                st.success(f"Estimated demand next month (Prophet): {next_month_pred} units (approx.)")

        # ---------- SARIMAX (ARIMA) ----------
        elif model_choice == "SARIMAX (ARIMA)":
            if not STATSMODELS_AVAILABLE:
                st.error("statsmodels is not installed in this environment. Install with `pip install statsmodels` and restart the app.")
            else:
                # Fit a simple SARIMAX (p,d,q) = (1,1,1) — you can tune these later
                try:
                    sarimax_order = (1, 1, 1)
                    seasonal_order = (0, 0, 0, 0)
                    model = SARIMAX(train_ts["Sales"].values, order=sarimax_order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
                    model_fit = model.fit(disp=False)

                    # Forecast for test + horizon
                    steps = len(test_ts) + horizon_months
                    forecast_res = model_fit.get_forecast(steps=steps)
                    forecast_values = forecast_res.predicted_mean

                    # MAE on holdout (first len(test_ts) predictions)
                    test_preds = forecast_values[:len(test_ts)]
                    mae = mean_absolute_error(test_ts["Sales"].values, test_preds)

                    # Build plot dataframe
                    history_dates = list(ts["Date"])
                    forecast_dates = [ts["Date"].max() + relativedelta(months=i) for i in range(1, steps + 1)]
                    combined_dates = history_dates + forecast_dates

                    combined_actual = list(ts["Sales"]) + [np.nan] * steps
                    combined_predicted = list(np.concatenate((np.full(len(ts), np.nan), forecast_values)))  # predicted aligned

                    plot_df = pd.DataFrame({
                        "Date": combined_dates,
                        "Actual": combined_actual,
                        "Predicted": combined_predicted
                    })

                    fig = px.line(plot_df, x="Date", y=["Actual", "Predicted"],
                                  labels={"value": "Sales", "variable": "Series"},
                                  title=f"SARIMAX Forecast (MAE on holdout: {mae:.2f})")
                    st.plotly_chart(fig, use_container_width=True)

                    st.success(f"Estimated demand next month (SARIMAX): {int(forecast_values[len(test_ts)])} units (approx.)")
                except Exception as e:
                    st.error(f"Failed to fit SARIMAX model: {e}")

# ---------- Data Table & download ----------
st.header("📋 Filtered Data")
st.dataframe(filtered_df)

csv = filtered_df.to_csv(index=False)
st.download_button(label="Download Filtered Data as CSV", data=csv, file_name="fashion_supply_data.csv", mime="text/csv")

# Footer
st.markdown("---")
st.markdown("""Dashboard built with Streamlit. Mock data used for demonstration — replace with real data (e.g., from ERP systems like SAP or SQL DB). 
For Prophet use: pip install prophet. For statsmodels/SARIMAX use: pip install statsmodels.""")
