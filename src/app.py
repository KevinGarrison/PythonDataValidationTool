import streamlit as st
import pandas as pd
from backend.utils import Utilitis

utils = Utilitis()

st.set_page_config(initial_sidebar_state='collapsed')

# Callback function
def set_page(page):
    if not st.session_state.data_cleaned:
        utils.non_numerical_data_cleaner()
        st.session_state.data_final = st.session_state.data_numerical
    if page == 'upload':
        st.session_state.algo_runned = False    
    st.session_state.page = page

if 'page' not in st.session_state:
    # Page state
    st.session_state.page = 'upload'
    # Data states
    st.session_state.data = None
    st.session_state.data_numerical = None
    st.session_state.data_standardized = None
    st.session_state.data_inversed = None
    st.session_state.data_filtered = None
    st.session_state.data_final = None
    # Stats state
    st.session_state.stats_summarize = None 
    # Filter range state
    st.session_state.filter_ranges = None
    st.session_state.filter_ranges_original = None
    # Action states
    st.session_state.data_cleaned = False
    st.session_state.upload_new_data = False
    st.session_state.standardized = False
    # Scaler parameters states
    st.session_state.scaler_mean = None
    st.session_state.scaler_scale = None
    st.session_state.method = None
    # TEST!!!
    st.session_state.data_test = None
    st.session_state.test_1 = ''
    
if st.session_state.page == 'upload': 
    st.title("Numerical Data Validation Tool")
    st.text("Upload your data below:")
    uploaded_file = st.file_uploader(label="Choose a file", type=['csv', 'xls', 'xlsx']) 
    if uploaded_file is not None:
        utils.load_data(uploaded_file=uploaded_file)
        st.subheader("Raw data:")
        st.write(st.session_state.data)
        st.button(label="Remove non-numerical data", on_click=set_page, args=['cleaned_data_page'])
        
if st.session_state.page == 'cleaned_data_page':
    st.subheader("Cleaned data (only numerical columns left):")
    st.write(st.session_state.data_final)
    method = st.selectbox(label='Choose method to determine feature ranges:',options=['Interquartil-Range-Method', 'Z-Score-Method', 'Advanced-Gamma-Method'])
    st.session_state.method = method
    st.page_link("pages/statistics.py", label="Data Statistics", icon="📊")
    st.page_link("pages/visualization.py", label="Data Visualization", icon="📈")
    st.page_link("pages/download.py", label="Determin feature ranges", icon="📐")   
    st.button("Upload new data", on_click=set_page, args=['upload'])
    

         