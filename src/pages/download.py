import streamlit as st
import pandas as pd
from backend.utils import Utilitis
from backend.stats import Statistics
import time

utils = Utilitis()
stats = Statistics()

st.header("Recommended Feature Ranges")

with st.spinner("Processing... Please wait."):
    method = st.session_state.method
    utils.run_algorithm(method)
    time.sleep(1)
st.success("Task completed!")

df = st.session_state.filter_ranges_original
for _, row in df.iterrows():
    datapoint = pd.DataFrame({'feature': [row['feature']], 'lower_bound': [row['lower_bound']], 'upper_bound': [row['upper_bound']]})
    df = st.session_state.data_numerical
    st.write(datapoint)
    stats.boxplot(df, row['feature'])

method = st.selectbox(label='Choose method to determine feature ranges:',options=['Interquartil-Range-Method', 'Z-Score-Method', 'Advanced-Gamma-Method'])
st.session_state.method = method
st.page_link("pages/statistics.py", label="Data Statistics", icon="📊")
st.page_link("pages/visualization.py", label="Data Visualization", icon="📈") # TODO implement download function
st.page_link("pages/download.py", label="Determin feature ranges", icon="📐")
st.page_link("app.py", label="Home", icon="🏠")