import pandas as pd
import streamlit as st
from dataclasses import dataclass

@dataclass
class Utilitis:
    
    @st.cache_data
    def non_numerical_data_cleaner(self, df:pd.DataFrame)->pd.DataFrame:
        return df.select_dtypes(include=['number'])
    
    @st.cache_data
    def load_data(self, uploaded_file) -> pd.DataFrame:
        if uploaded_file is not None:
            file_path = str(uploaded_file.name)
            if file_path[-4:] == ".csv":
                st.session_state.data = pd.read_csv(uploaded_file)
            elif (file_path[-4:] == ".xls") | (file_path[-4:] == ".xlsx"):
                st.session_state.data = pd.read_excel(uploaded_file)
            else:
                raise TypeError(f"The file has a wrong type {file_path.name}")
        else:
            raise ValueError("No file uploaded") 
        
    def session_state_data_clearer(self):
        st.session_state.data = None

    def session_state_clearer(self):
        st.session_state.clear()
