import pandas as pd, numpy as np
import streamlit as st
from dataclasses import dataclass
from backend.stats import Statistics
import openpyxl


stats = Statistics()

@dataclass
class Utilitis:

    @st.cache_data
    def load_data(self, uploaded_file) -> pd.DataFrame:
        '''loads the data format csv or excel into a pandas dataframe'''
        if uploaded_file is not None:
            file_path = str(uploaded_file.name)
            if file_path.endswith(".csv"):
                return pd.read_csv(uploaded_file)
            elif (file_path.endswith(".xls")) | (file_path.endswith(".xlsx")):
                return pd.read_excel(uploaded_file)
            else:
                raise TypeError(f"The file has a wrong type {file_path.name}")
                
        else:
            raise ValueError("No file uploaded")
            
    

    @st.cache_data
    def session_state_clearer(self):
        '''clears all session states'''
        st.session_state.clear()


    @st.cache_data
    def run_algorithm(self, df:pd.DataFrame, approach:str):
        '''runs the chosen algorithm for outlier detection'''
        try:
            df = df
            if approach == 'Interquartil-Range-Method':
                ranges = stats.iqr_approach(df)
            elif approach == 'Standard-Deviation-Method':
                ranges = stats.std_approach(df)
            elif approach == 'Modified-Z-Score-Method':
                ranges = stats.modified_z_score_approach(df)
            elif approach == 'Advanced-Gamma-Method':
                ranges = stats.gamma_method_modified(df)
            return ranges
        except:
            print('Algorithm failed')
            return None
        

    @st.cache_data
    def apply_filters_on_features(self):
        if 'filter_ranges' in st.session_state:
            filter_ranges_df = st.session_state.filter_ranges
            df = st.session_state.data_standardized
            for feature in filter_ranges_df['feature']:
                lower = filter_ranges_df.loc[filter_ranges_df['feature'] == feature, 'lower_bound'].values[0]
                upper = filter_ranges_df.loc[filter_ranges_df['feature'] == feature, 'upper_bound'].values[0]

                mask = (df[feature] > lower) & (df[feature] < upper)
                df[feature] = df[feature][mask]
                st.session_state.data_filtered = None
                st.session_state.data_filtered = df

    
