import pandas as pd
import streamlit as st
from dataclasses import dataclass
from backend.stats import Statistics
import great_expectations as gx


stats = Statistics()

@dataclass
class Utilitis:

    @st.cache_data
    def non_numerical_data_cleaner(self):
        '''removes all columns that are not numerical'''
        df = st.session_state.data
        st.session_state.data_numerical =  df.select_dtypes(include=['number'])

    @st.cache_data
    def load_data(self, uploaded_file) -> pd.DataFrame:
        '''loads the data format csv or excel into a pandas dataframe'''
        if uploaded_file is not None:
            file_path = str(uploaded_file.name)
            if file_path.endswith(".csv"):
                st.session_state.data = pd.read_csv(uploaded_file)
            elif (file_path.endswith(".xls")) | (file_path.endswith(".xlsx")):
                st.session_state.data = pd.read_excel(uploaded_file)
            else:
                raise TypeError(f"The file has a wrong type {file_path.name}")
        else:
            raise ValueError("No file uploaded") 
    
    @st.cache_data
    def session_state_data_clearer(self):
        '''clears all data in session states'''
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
        # Action states
        st.session_state.data_cleaned = False
        st.session_state.upload_new_data = False
        st.session_state.algo_runned = False
        st.session_state.standardized = False
        # Scaler parameters states
        st.session_state.scaler_mean = None
        st.session_state.scaler_scale = None
        

    @st.cache_data
    def session_state_clearer(self):
        '''clears all session states'''
        st.session_state.clear()

    @st.cache_data
    def run_algorithm(self, approach:str):  # TODO implement the parameters via user input
        '''runs the chosen algorithm for outlier detection'''
        try:
            st.session_state.data_filtered = None
            stats.normalize_numerical_data()
            if approach == 'Interquartil-Range-Method':
                stats.iqr_approach()
            elif approach == 'Z-Score-Method':
                stats.std_approach()
            elif approach == 'Advanced-Gamma-Method':
                stats.gamma_outlier()
            #self.apply_filters_on_features() # TODO call function mabey in another part of the code if needed
            #stats.inverse_normalize_data()
            stats.inverse_filter_ranges()
        except:
            print('Algorithm failed')
        
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

                st.session_state.data_filtered = df

    @st.cache_data
    def setup_gx(self):
        '''Set up Great Expectations context and connect to the loaded data'''
        # Initialize Great Expectations context
        context = gx.get_context()
        data_source_name = "numerical_eval_data_source"
        context.data_sources.add_pandas(name=data_source_name)
        data_source = context.data_sources.get(data_source_name)
        data_asset_name = "numerical_eval_data_asset"
        data_source.add_dataframe_asset(name=data_asset_name)
        data_asset = context.data_sources.get(data_source_name).get_asset(data_asset_name)
        batch_definition_name = "numerical_eval_batch_definition"
        st.session_state.batch_definition = data_asset.add_batch_definition_whole_dataframe(
           batch_definition_name
        )

    @st.cache_data
    def define_z_score_exp(self, column):
        '''Define an expectation for column z-scores within a threshold'''
        # Add an expectation that z-scores for the specified column are within a threshold
        st.session_state.z_score_ex = gx.expectations.ExpectColumnValueZScoresToBeLessThan(
            column=column, threshold=3, double_sided=True
        )
    
    @st.cache_data
    def define_column_values_between_exp(self, column, min, max):
        '''Define an expectation for column mean values to be between a specified range'''
        # Add an expectation for the column's mean value to be between specified min and max
        st.session_state.min_max_ex = gx.expectations.ExpectColumnValuesToBeBetween(
            column=column, min_value=min, max_value=max
        )