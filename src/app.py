import streamlit as st
import pandas as pd

st.title("Data Validation Tool")

st.text("Upload your data below:")

uploaded_file = st.file_uploader(label="Choose a file", type=['csv', 'xls', 'xlsx'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file) # TODO: Change this hard coded method to a data_loader Object
    st.write(df)

