import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Universal Analytics System",
    page_icon="📦",
    layout="wide"
)

st.sidebar.success("Select a page from the sidebar")

st.title("📦 Universal Analytics System")
st.markdown(
    """
    Welcome to the **Universal Supply Chain Analytics Dashboard**.

    - Upload any CSV dataset  
    - Explore KPIs  
    - Generate charts  
    - Forecast trends  
    - Predict demand with AI  
    - All pages auto-adapt to your dataset  
    """
)

uploaded_file = st.file_uploader("Upload a CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.session_state["df"] = df
    st.success("Dataset uploaded successfully!")
    st.dataframe(df.head())
else:
    st.warning("Please upload a dataset to continue.")
