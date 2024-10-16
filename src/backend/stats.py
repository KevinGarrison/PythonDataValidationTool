import pandas as pd
import streamlit as st
from dataclasses import dataclass 

@dataclass
class Statistics: # TODO: implement the class
    
    @st.cache_data
    def feature_stats_summarize(df:pd.DataFrame)->pd.DataFrame:
        return df.describe()
        
    @st.cache_data
    def calc_stats(df:pd.DataFrame)->dict:
        st.session_state.df_stats_dict = {}

        st.session_state.num_rows = df.shape[0]    
        st.session_state.num_cols = df.shape[1]

        if len(df.columns) > 1:
            st.session_state.num_missing_values = df.isnull().sum().sum()
            
        else:
            st.session_state.mean = df.mean()
            st.session_state.median = df.median()
            st.session_state.mode = df.mod()
            st.session_state.std = df.std()
            st.session_state.var = st.session_state.std**2
            st.session_state.num_missing_values = df.isnull().sum()
            st.session_state.normal_dist = (st.session_state.mean == st.session_state.median == st.session_state.mode)
            st.session_state.left_skewed = (st.session_state.mean < st.session_state.median)
            st.session_state.right_skewed = (st.session_state.mean > st.session_state.median)
        



