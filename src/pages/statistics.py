import streamlit as st
import pandas as pd

st.header("Statistics of your data")

data = st.session_state.data

selections =  ["All data"] + list(data.columns)
# Dropdown to select the feature to plot
feature = st.selectbox('Select a feature to get the statistics:', selections)

if feature == "All data":
    description = data.describe()
    st.write(description)
if feature in list(data.columns):
    description = data.describe() 
    st.write(description[feature])
st.page_link("pages/visualization.py", label="Data Visualization", icon="📈")
st.page_link("app.py", label="Home", icon="🏠")



