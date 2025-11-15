# app_local.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Flight Price Prediction App", layout="wide")

st.title("✈️ Flight Price Analysis Dashboard")
st.markdown("This app uses the local file **Data_Train.xlsx** for analysis and modeling.")

# Load data
@st.cache_data
def load_data():
    data = pd.read_excel("Data_Train.xlsx")
    return data

data = load_data()

st.subheader("Dataset Preview")
st.dataframe(data.head())

st.subheader("Basic Info")
st.write("Shape:", data.shape)
st.write("Missing values:")
st.write(data.isnull().sum())

# Visualizations
st.subheader("Visualizations")

num_cols = data.select_dtypes(include=np.number).columns.tolist()
cat_cols = data.select_dtypes(exclude=np.number).columns.tolist()

option = st.selectbox("Choose a plot type", ["Histogram", "Boxplot", "Correlation Heatmap"])

if option == "Histogram":
    column = st.selectbox("Select numerical column", num_cols)
    fig, ax = plt.subplots()
    sns.histplot(data[column], kde=True, ax=ax)
    st.pyplot(fig)

elif option == "Boxplot":
    column = st.selectbox("Select numerical column", num_cols)
    fig, ax = plt.subplots()
    sns.boxplot(data[column], ax=ax)
    st.pyplot(fig)

elif option == "Correlation Heatmap":
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

st.success("✅ Analysis complete!")
