import streamlit as st
import pandas as pd
from backend.data_loader import DataLoader

st.set_page_config(initial_sidebar_state='collapsed')

def reset_app():
     st.session_state.clear()

if 'page' not in st.session_state:
        st.session_state.page = 'upload'

if st.session_state.page == 'upload':
    st.title("Data Validation Tool")
    st.text("Upload your data below:")
    uploaded_file = st.file_uploader(label="Choose a file", type=['csv', 'xls', 'xlsx']) 
    if uploaded_file is not None:
        loader = DataLoader()
        loader.load_data(uploaded_file)
        st.session_state.data = loader.df
        st.write(st.session_state.data)
        st.page_link("pages/statistics.py", label="Data Statistics", icon="1️⃣")
        st.page_link("pages/visualization.py", label="Data Visualization", icon="2️⃣")
        
    

