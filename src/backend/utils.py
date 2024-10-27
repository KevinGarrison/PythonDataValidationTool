import pandas as pd
import streamlit as st
from dataclasses import dataclass
from backend.stats import Statistics


stats = Statistics()

@dataclass
class Utilitis:
    @st.cache_data
    def non_numerical_data_cleaner(self):
        df = st.session_state.data
        st.session_state.data =  df.select_dtypes(include=['number'])
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
    @st.cache_data    
    def session_state_data_clearer(self):
        st.session_state.data = None
    @st.cache_data
    def session_state_clearer(self):
        st.session_state.clear()
    @st.cache_data
    def run_algorithm(self, approach:str='GAMMA'):  # TODO implement the parameters via user input
        try:
            df = st.session_state.data
            stats.normalize_numerical_data()

            if approach == 'IQR':
                stats.iqr_approach()
            elif approach == 'STD':
                stats.std_approach()
            elif approach =='GAMMA':
                stats.gamma_outlier() 
            st.session_state.algo_runned = True
        except:
            print('Algorithm failed')
        
    @st.cache_data
    def apply_filters_on_features(self):
        if 'filter_ranges' in st.session_state:

            filter_ranges_df = st.session_state.filter_ranges
            df = st.session_state.data

            for feature in filter_ranges_df['feature']:
                lower = filter_ranges_df.loc[filter_ranges_df['feature'] == feature, 'lower_bound'].values[0]
                upper = filter_ranges_df.loc[filter_ranges_df['feature'] == feature, 'upper_bound'].values[0]

                # Apply the filter and create a new Series with only the valid values
                filtered_values = df[feature][(df[feature] > lower) & (df[feature] < upper)]

                # Assign filtered values back to the DataFrame
                df[feature] = filtered_values

                st.session_state.data = df
