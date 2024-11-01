import pandas as pd, numpy as np
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
    def run_algorithm(self, df, approach:str):
        '''runs the chosen algorithm for outlier detection'''
        try:
            df = df
            if approach == 'Interquartil-Range-Method':
                ranges = stats.iqr_approach(df)
            elif approach == 'Z-Score-Method':
                ranges = stats.z_score_approach(df)
            elif approach == 'Modified-Z-Score-Method':
                ranges = stats.modified_z_score_approach(df)
            elif approach == 'Advanced-Gamma-Method':
                df_normalized = stats.normalize_numerical_data(df)
                ranges_normalized = stats.gamma_outlier(df_normalized)
                return stats.inverse_filter_ranges(ranges_normalized)
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

    @st.cache_data
    def setup_gx(self):
        '''Set up Great Expectations context and connect to the loaded data'''
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
    def define_column_values_between_exp(self, column, min, max):
        '''Define an expectation for column mean values to be between a specified range'''
        return gx.expectations.ExpectColumnValuesToBeBetween(
            column=column, min_value=min, max_value=max
        )
    
    def create_range_sliders(self, data, lower, upper, alpha=2):
        # Calculate mean and standard deviation
        std_value = data.std()

        # Calculate the limits for the sliders
        limit_lower_1 = lower - alpha * std_value
        limit_upper_1 = lower + alpha * std_value

        limit_lower_2 = upper - alpha * std_value
        limit_upper_2 = upper + alpha * std_value

        # Create a slider for the upper bound
        lower_bound = st.slider(
            "Adjust Lower Bound:",
            min_value=limit_lower_1,  # Set minimum to lower_limit
            max_value=limit_upper_1,  # Set maximum to upper_limit
            value=lower,  # Default value is the mean
            step=0.1  # Adjust this based on your data precision
        )

        # Create a slider for the lower bound
        upper_bound = st.slider(
            "Adjust Upper Bound:",
            min_value=limit_lower_2,  # Set minimum to lower_limit
            max_value=limit_upper_2,  # Set maximum to upper_limit
            value=upper,  # Default value is mean - std
            step=0.1  # Adjust this based on your data precision
        )

        return lower_bound, upper_bound