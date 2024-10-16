import streamlit as st
import pandas as pd
from backend.data_loader import DataLoader
from backend.utils import Utilitis

st.set_page_config(initial_sidebar_state='collapsed')

def reset_app():
    st.session_state.clear()
# Callback function
def set_page(page):
     if not st.session_state.data_cleaned:
        st.session_state.data = utils.non_numerical_data_cleaner(st.session_state.data) # TODO fix bug with cleaned data visualizing
     st.session_state.page = page

if 'page' not in st.session_state:
    st.session_state.page = 'upload'
    st.session_state.data_cleaned = False
    st.session_state.upload_new_data = False
    
if st.session_state.page == 'upload':
    st.title("Numerical Data Validation Tool")
    st.text("Upload your data below:")
    uploaded_file = st.file_uploader(label="Choose a file", type=['csv', 'xls', 'xlsx']) 
    if uploaded_file is not None:
        loader = DataLoader()
        loader.load_data(uploaded_file)
        utils = Utilitis()
        st.session_state.data = loader.df
        st.subheader("Raw data:")
        st.write(st.session_state.data)

        st.button(label="Remove non-numerical data", on_click=set_page, args=['cleaned_data_page'])
        
            

if st.session_state.page == 'cleaned_data_page':
    st.subheader("Cleaned data (only numerical columns left):")
    st.write(st.session_state.data)
    st.write("Choose an option below:")
    st.page_link("pages/statistics.py", label="Data Statistics", icon="📊")
    st.page_link("pages/visualization.py", label="Data Visualization", icon="📈")
    st.button("Upload new data", on_click=set_page, args=['upload'])

         