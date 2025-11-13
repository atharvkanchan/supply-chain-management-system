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
st.markdown("<div class='footer'>Built with ❤️ — Advanced Supply Chain Analytics</div>", unsafe_allow_html=True)
