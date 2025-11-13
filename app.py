Below are the upgraded project files that fully incorporate the additional fields you provided and implement a richer dashboard: KPIs, interactive filters, many charts, cost breakdowns, lead-time analysis, carbon & fuel analytics, damage/return analysis, regression, and a simple ML forecasting panel.

Place these files in your project folder and run: `streamlit run app.py`.

---

# app.py

```python
# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import load_data, preprocess_data, aggregate_weekly
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from datetime import datetime

st.set_page_config(page_title="Supply Chain Analytics — Advanced", layout="wide", page_icon="📦")

# --- custom styles ---
with open("assets/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- header ---
col1, col2 = st.columns([0.12, 0.88])
with col1:
    st.image("assets/logo.png", width=80)
with col2:
    st.title("Supply Chain Management — Advanced Analytics")
    st.markdown("_Interactive dashboard: costs, logistics, sustainability & forecasting_")

# --- load data ---
@st.cache_data
def get_data(path="data/india_supply_chain_2024_2025.csv"):
    df = load_data(path)
    df = preprocess_data(df)
    return df

try:
    df = get_data()
except FileNotFoundError:
    st.error("Dataset not found at data/india_supply_chain_2024_2025.csv. Place your file there or update the path in app.py")
    st.stop()

# --- Sidebar filters ---
st.sidebar.header("Filters")
# dates
min_date = df['date'].min()
max_date = df['date'].max()
date_range = st.sidebar.date_input("Date range", [min_date, max_date], min_value=min_date, max_value=max_date)

# region/supplier/product if present
regions = st.sidebar.multiselect("Region", options=sorted(df['region'].unique()), default=sorted(df['region'].unique()))
suppliers = st.sidebar.multiselect("Supplier", options=sorted(df['supplier'].unique()), default=sorted(df['supplier'].unique()))
products = st.sidebar.multiselect("Product Category", options=sorted(df['product_category'].unique()), default=sorted(df['product_category'].unique()))

# numeric sliders
distance_rng = st.sidebar.slider("Distance (km)", float(df['distance_km'].min()), float(df['distance_km'].max()), (float(df['distance_km'].min()), float(df['distance_km'].max())))
price_rng = st.sidebar.slider("Price per unit (INR)", float(df['price_per_unit_inr'].min()), float(df['price_per_unit_inr'].max()), (float(df['price_per_unit_inr'].min()), float(df['price_per_unit_inr'].max())))

st.sidebar.markdown("---")
if st.sidebar.button("Reset filters"):
    st.experimental_rerun()

# --- Apply filters ---
start_dt, end_dt = date_range
mask = (
    (df['date'] >= pd.to_datetime(start_dt)) &
    (df['date'] <= pd.to_datetime(end_dt)) &
    (df['region'].isin(regions)) &
    (df['supplier'].isin(suppliers)) &
    (df['product_category'].isin(products)) &
    (df['distance_km'] >= distance_rng[0]) &
    (df['distance_km'] <= distance_rng[1]) &
    (df['price_per_unit_inr'] >= price_rng[0]) &
    (df['price_per_unit_inr'] <= price_rng[1])
)

filtered = df.loc[mask].copy()
if filtered.empty:
    st.warning("No data for selected filters — please broaden selection.")

# --- KPI row ---
st.subheader("Key Performance Indicators")
k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    total_orders = int(filtered.shape[0])
    st.metric("Total Shipments", f"{total_orders}")
with k2:
    total_units_ordered = int(filtered['units_ordered'].sum())
    total_units_shipped = int(filtered['units_shipped'].sum())
    st.metric("Units Ordered / Shipped", f"{total_units_ordered} / {total_units_shipped}")
with k3:
    total_cost = filtered['total_cost_inr'].sum()
    st.metric("Total Cost (INR)", f"{total_cost:,.0f}")
with k4:
    transport_cost = filtered['transport_cost_inr'].sum()
    st.metric("Transport Cost (INR)", f"{transport_cost:,.0f}")
with k5:
    gst_amount = filtered['gst_amount_inr'].sum()
    st.metric("GST Amount (INR)", f"{gst_amount:,.0f}")

k6, k7, k8, k9 = st.columns(4)
with k6:
    avg_lead_planned = filtered['planned_lead_days'].mean()
    avg_lead_actual = filtered['actual_lead_days'].mean()
    st.metric("Lead Time (Planned vs Actual)", f"{avg_lead_planned:.1f} / {avg_lead_actual:.1f} days")
with k7:
    on_time_pct = filtered['on_time_delivery'].mean() * 100
    st.metric("On-time Delivery (%)", f"{on_time_pct:.1f}%")
with k8:
    damage_rate = filtered['damage_rate'].mean() * 100
    st.metric("Damage Rate (%)", f"{damage_rate:.2f}%")
with k9:
    total_carbon = filtered['carbon_kg'].sum()
    st.metric("Total Carbon (kg)", f"{total_carbon:,.0f}")

st.markdown("---")

# --- Main layout: Trends / Breakdown / Details ---
left, middle, right = st.columns([2,2,1])

with left:
    st.subheader("Demand Forecast vs Actual Sales")
    df_ts = filtered.groupby('date').agg(demand_forecast=('demand_forecast','sum'), actual_sales=('actual_sales','sum')).reset_index()
    if not df_ts.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_ts['date'], y=df_ts['demand_forecast'], mode='lines+markers', name='Demand Forecast'))
        fig.add_trace(go.Scatter(x=df_ts['date'], y=df_ts['actual_sales'], mode='lines+markers', name='Actual Sales'))
        fig.update_layout(legend=dict(orientation='h'))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Units Ordered vs Units Shipped")
    df_units = filtered.groupby('date').agg(units_ordered=('units_ordered','sum'), units_shipped=('units_shipped','sum')).reset_index()
    if not df_units.empty:
        fig2 = px.bar(df_units.melt(id_vars='date', value_vars=['units_ordered','units_shipped']), x='date', y='value', color='variable', barmode='group', title='Daily Units Ordered vs Shipped')
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Lead Time Analysis")
    if not filtered.empty:
        fig3 = px.box(filtered, x='product_category', y='actual_lead_days', points='outliers', title='Actual Lead Days by Product Category')
        st.plotly_chart(fig3, use_container_width=True)

with middle:
    st.subheader("Cost Breakdown")
    cost_cols = ['cost_per_unit_inr','transport_cost_inr','toll_charges_inr','gst_amount_inr']
    cost_summary = filtered[cost_cols].sum().reset_index()
    cost_summary.columns = ['cost_component','amount']
    if not cost_summary.empty:
        fig4 = px.pie(cost_summary, names='cost_component', values='amount', title='Cost Breakdown')
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Distance vs Transport Cost")
    if not filtered.empty:
        fig5 = px.scatter(filtered, x='distance_km', y='transport_cost_inr', trendline='ols', hover_data=['supplier','region','units_shipped'])
        st.plotly_chart(fig5, use_container_width=True)

    st.subheader("Sustainability & Fuel")
    fuel_ts = filtered.groupby('date').agg(avg_fuel_price=('fuel_price_per_litre_inr','mean'), carbon_kg=('carbon_kg','sum')).reset_index()
    if not fuel_ts.empty:
        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(x=fuel_ts['date'], y=fuel_ts['avg_fuel_price'], name='Avg Fuel Price (INR/L)'))
        fig6.add_trace(go.Bar(x=fuel_ts['date'], y=fuel_ts['carbon_kg'], name='Carbon (kg)', yaxis='y2', opacity=0.6))
        fig6.update_layout(yaxis2=dict(overlaying='y', side='right', title='Carbon (kg)'), legend=dict(orientation='h'))
        st.plotly_chart(fig6, use_container_width=True)

with right:
    st.subheader("Top 10 Delayed Shipments")
    if 'delay_days' in filtered.columns:
        delayed = filtered.sort_values('delay_days', ascending=False).head(10)
        st.dataframe(delayed[['shipment_id','supplier','region','delay_days','actual_lead_days','on_time_delivery']].reset_index(drop=True))
    else:
        st.info('No delay_days column present')

    st.markdown('---')
    st.subheader('Damage & Return Overview')
    dmg = filtered[['damage_rate','return_rate']].mean().rename({'damage_rate':'avg_damage_rate','return_rate':'avg_return_rate'})
    st.write(dmg)

    st.markdown('---')
    st.subheader('Download & Export')
    st.download_button('Download filtered CSV', filtered.to_csv(index=False).encode('utf-8'), file_name='filtered_supply_chain.csv')

st.markdown('---')

# --- Detailed Table and Aggregations ---
st.subheader('Aggregated Weekly Summary')
weekly = aggregate_weekly(filtered)
st.dataframe(weekly, use_container_width=True)

st.markdown('---')

# --- Simple ML Forecasting panel: Forecast actual_sales using past features ---
st.subheader('Quick Forecast: Predict next-day Actual Sales (RandomForest)')
if st.button('Run forecast'):
    model_df = filtered.copy()
    # prepare features: numeric columns and lagged sales
    model_df = model_df.sort_values('date')
    model_df['sales_lag_1'] = model_df['actual_sales'].shift(1).fillna(method='bfill')
    features = ['demand_forecast','units_ordered','units_shipped','price_per_unit_inr','cost_per_unit_inr','transport_cost_inr','distance_km','sales_lag_1']
    model_df = model_df.dropna(subset=['actual_sales'])
    X = model_df[features].fillna(0)
    y = model_df['actual_sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    st.write(f'Mean Absolute Error on holdout: {mae:.2f} units')
    # show feature importances
    fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    st.bar_chart(fi)

st.markdown('---')
st.subheader('Raw Data')
st.dataframe(filtered.reset_index(drop=True), use_container_width=True)

st.caption(f'Data range: {min_date.date()} — {max_date.date()} | Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

# footer
st.markdown("<div class='footer'>Built with ❤️ — Advanced Supply Chain Analytics</div>", unsafe_allow_html=True)
```

---

# utils.py

```python
# utils.py
import pandas as pd

def load_data(path: str):
    df = pd.read_csv(path)
    return df


def preprocess_data(df: pd.DataFrame):
    # normalize column names
    df = df.rename(columns=lambda x: x.strip().lower().replace(' ', '_'))

    # date parsing: try common names
    for col in ['date','shipment_date','created_at']:
        if col in df.columns:
            df['date'] = pd.to_datetime(df[col])
            break
    if 'date' not in df.columnsmns:
        raise FileNotFoundError('No date column found; add `date` or `shipment_date` to CSV')

    # ensure numeric columns exist and are numeric
    numeric_cols = ['distance_km','demand_forecast','actual_sales','units_ordered','units_shipped',
                    'price_per_unit_inr','cost_per_unit_inr','transport_cost_inr','toll_charges_inr',
                    'gst_percent','gst_amount_inr','total_cost_inr','planned_lead_days','actual_lead_days',
                    'on_time_delivery','delay_days','damage_rate','return_rate','fuel_price_per_litre_inr','carbon_kg']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            # fill missing numeric columns with zeros or sensible NA
            if c.startswith('on_time'):
                df[c] = 0
            else:
                df[c] = 0

    # normalize boolean on_time_delivery to 0/1
    if df['on_time_delivery'].dtype == object:
        df['on_time_delivery'] = df['on_time_delivery'].map({'yes':1,'y':1,'true':1,'True':1,'1':1}).fillna(0)

    # ensure categorical columns
    for c in ['region','supplier','product_category','shipment_id']:
        if c not in df.columns:
            df[c] = 'Unknown' if c != 'shipment_id' else range(1, len(df)+1)

    return df


def aggregate_weekly(df: pd.DataFrame):
    # weekly aggregates
    w = df.set_index('date').resample('W').agg({
        'actual_sales':'sum',
        'demand_forecast':'sum',
        'units_ordered':'sum',
        'units_shipped':'sum',
        'total_cost_inr':'sum',
        'transport_cost_inr':'sum',
        'carbon_kg':'sum',
        'fuel_price_per_litre_inr':'mean'
    }).reset_index()
    return w
```

---

# assets/styles.css

```css
:root{ --accent:#BFA67A; --muted:#6b7280; }
.stApp{ background-color: #ffffff; }
h1,h2,h3{ font-family: 'Inter', sans-serif; }
.footer{ text-align:center; color:var(--muted); margin-top:18px }
[data-testid="stSidebar"]{ width: 320px; }
```

---

# requirements.txt

```
streamlit>=1.20
pandas
numpy
plotly
scikit-learn
```

---

# README.md

```md
# Advanced Supply Chain Dashboard

## Files
- `app.py` - main Streamlit app
- `utils.py` - helpers: load, preprocess, aggregate
- `assets/styles.css` - styling
- `assets/logo.png` - optional logo
- `data/india_supply_chain_2024_2025.csv` - your CSV (already uploaded)

## Run
1. create venv & install requirements
2. place dataset at `data/india_supply_chain_2024_2025.csv` (you have uploaded this file)
3. `streamlit run app.py`

## Notes
- The app expects the numeric fields you listed. `utils.py` will coerce missing numeric fields to zero; adapt as needed.
- Forecast panel uses a simple RandomForest on available features; you can replace with time-series library for better accuracy.

```

---

If you want, I will now:

* 1. add a small synthetic-data fallback so the app runs even if your CSV path differs, or
* 2. adapt the app to any column name mismatches automatically by inspecting your uploaded CSV now.

I went ahead and assumed your uploaded CSV is named `data/india_supply_chain_2024_2025.csv` (you uploaded `/mnt/data/india_supply_chain_2024_2025.csv`). If you'd prefer a different path or want me to auto-inspect the CSV and adapt column-name mapping, tell me and I will update the files accordingly.
.footer{ text-align:center; color:var(--muted); margin-top:18px }
