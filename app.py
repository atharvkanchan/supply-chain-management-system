import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Set page config
st.set_page_config(page_title="Fashion Supply Management Analytics", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Inventory Analysis", "Demand Forecasting", "Supplier Performance"])

# Sample data for demonstration
@st.cache_data
def load_sample_data():
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    data = pd.DataFrame({
        "Date": dates,
        "Sales": np.random.randint(100, 500, 100),
        "Inventory_Level": np.random.randint(50, 200, 100),
        "Supplier_Delivery_Time": np.random.randint(1, 10, 100),
        "Demand": np.random.randint(80, 450, 100),
        "Product": np.random.choice(["Dress", "Shoes", "Bag", "Jacket"], 100)
    })
    return data

sample_data = load_sample_data()

# Function to upload data
def upload_data():
    uploaded_file = st.file_uploader("Upload your CSV file (columns: Date, Sales, Inventory_Level, Supplier_Delivery_Time, Demand, Product)", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        data["Date"] = pd.to_datetime(data["Date"])
        return data
    else:
        return sample_data

# Dashboard Page
if page == "Dashboard":
    st.title("Fashion Supply Management Analytics Dashboard")
    data = upload_data()
    
    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Sales", f"${data['Sales'].sum():,}")
    col2.metric("Avg Inventory Level", f"{data['Inventory_Level'].mean():.0f}")
    col3.metric("Avg Delivery Time", f"{data['Supplier_Delivery_Time'].mean():.1f} days")
    col4.metric("Total Demand", f"{data['Demand'].sum():,}")
    
    st.subheader("Sales Trend Over Time")
    fig = px.line(data, x="Date", y="Sales", title="Sales Trend")
    st.plotly_chart(fig, use_container_width=True)

# Inventory Analysis Page
elif page == "Inventory Analysis":
    st.title("Inventory Analysis")
    data = upload_data()
    
    st.subheader("Inventory Levels by Product")
    inventory_by_product = data.groupby("Product")["Inventory_Level"].mean().reset_index()
    fig = px.bar(inventory_by_product, x="Product", y="Inventory_Level", title="Average Inventory by Product")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Inventory vs. Sales Scatter Plot")
    fig = px.scatter(data, x="Inventory_Level", y="Sales", color="Product", title="Inventory vs. Sales")
    st.plotly_chart(fig, use_container_width=True)

# Demand Forecasting Page
elif page == "Demand Forecasting":
    st.title("Demand Forecasting")
    data = upload_data()
    
    # Simple linear regression for forecasting
    X = data[["Sales", "Inventory_Level"]]
    y = data["Demand"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    st.subheader("Forecasted Demand")
    predictions = model.predict(X_test)
    forecast_df = pd.DataFrame({"Actual": y_test, "Predicted": predictions})
    st.dataframe(forecast_df.head(10))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["Actual"], mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["Predicted"], mode="lines", name="Predicted"))
    fig.update_layout(title="Demand Forecasting: Actual vs. Predicted")
    st.plotly_chart(fig, use_container_width=True)

# Supplier Performance Page
elif page == "Supplier Performance":
    st.title("Supplier Performance")
    data = upload_data()
    
    st.subheader("Delivery Time Distribution")
    fig = px.histogram(data, x="Supplier_Delivery_Time", title="Supplier Delivery Time Distribution")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Average Delivery Time by Product")
    delivery_by_product = data.groupby("Product")["Supplier_Delivery_Time"].mean().reset_index()
    fig = px.bar(delivery_by_product, x="Product", y="Supplier_Delivery_Time", title="Avg Delivery Time by Product")
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with Streamlit for Fashion Supply Analytics")



