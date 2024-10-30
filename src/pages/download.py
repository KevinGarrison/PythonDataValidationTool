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
    utils.setup_gx()
    time.sleep(1)
st.success("Task completed!")

df = st.session_state.filter_ranges_original

batch_parameters = {"dataframe": st.session_state.data}
batch = st.session_state.batch_definition.get_batch(
    batch_parameters=batch_parameters
)        

for _, row in df.iterrows():
    feature = row['feature']
    lower = row['lower_bound']
    upper = row['upper_bound']
    datapoint = pd.DataFrame({'feature': [feature], 'lower_bound': [lower], 'upper_bound': [upper]})
    df = st.session_state.data_numerical
    st.write(datapoint)
    stats.boxplot_px(df, row['feature'])
    utils.define_column_values_between_exp(column=feature, min=lower, max=upper)
    expectation_min_max = st.session_state.min_max_ex
    validation_result = batch.validate(expectation_min_max)
    st.write("Validation Results:", validation_result)

method = st.selectbox(label='Choose method to determine feature ranges:',options=['Interquartil-Range-Method', 'Z-Score-Method', 'Advanced-Gamma-Method'])
st.session_state.method = method
st.page_link("pages/statistics.py", label="Data Statistics", icon="📊")
st.page_link("pages/visualization.py", label="Data Visualization", icon="📈") # TODO implement download function
st.page_link("pages/download.py", label="Determin feature ranges", icon="📐")
st.page_link("app.py", label="Home", icon="🏠")