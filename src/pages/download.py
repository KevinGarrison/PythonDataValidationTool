import streamlit as st
from backend.utils import Utilitis

utils = Utilitis()

st.header("Outliers removed of your data")

utils.run_algorithm()

utils.apply_filters_on_features()

df = st.session_state.data

selections =  ["All data"] + list(df.columns)
# Dropdown to select the feature to plot
feature = st.selectbox('Select a feature to get the statistics:', selections)

if feature == "All data":
    st.write(df)
if feature in list(df.columns): 
    st.write(df[feature])
st.page_link("pages/statistics.py", label="Data Statistics", icon="📊")
st.page_link("pages/visualization.py", label="Data Visualization", icon="📈") # TODO implement download function
st.page_link("app.py", label="Home", icon="🏠")