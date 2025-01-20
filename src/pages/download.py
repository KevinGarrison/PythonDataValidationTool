import streamlit as st
import pandas as pd
from backend.utils import Utilitis
from backend.stats import Statistics
from app import update_method
import time

st.header("Recommended Feature Ranges", divider="rainbow")

data = st.session_state.data

if not data.empty:

    utils = Utilitis()
    stats = Statistics()

    with st.spinner("Processing... Please wait."):
        method = st.session_state.selected_method
        original_ranges = utils.run_algorithm(data, method)
        st.write('Results Overview:',)
        st.write(original_ranges,)
        data_collection = dict()



        for index, row in original_ranges.iterrows():
            feature = row['feature']
            lower = row['lower_bound']
            upper = row['upper_bound']
            data_collection[feature] = dict()
            data_collection[feature]['lower'] = lower 
            data_collection[feature]['upper'] = upper 
            data_collection[feature]['total_count'] = data[feature].notna().sum()
            masked_values = (data[feature] < lower) | (data[feature] > upper)
            data_collection[feature]['not_valid_count'] = masked_values.sum()
            percentage_outside = (data_collection[feature]['not_valid_count'] / data_collection[feature]['total_count']) * 100
            data_collection[feature]['not_valid_perc'] = round(percentage_outside, 2)
            

        selected_feature = st.selectbox(label="Choose a feature to get expected value range and visualisation:",options=list(original_ranges['feature']))

        df_num = st.session_state.data

        if selected_feature in list(df_num.columns):
            if data_collection[selected_feature]['not_valid_count'] == 0:
                st.write(f'Data range for **{selected_feature}** meets expectations with approach:',)
                st.write(f"{st.session_state.selected_method}",)
            else:
                st.write(f'Data range for **{selected_feature}** is not as expected with approach:',)
                st.write(f"{st.session_state.selected_method}",)
            stats.boxplot_px(df_num, original_ranges, selected_feature)
            
            
            lower = data_collection[selected_feature]['lower']
            upper = data_collection[selected_feature]['upper']
            st.write(f'Expected lower bound: {lower}',)
            st.write(f'Expected upper bound: {upper}',)
            st.write(f'Observations total: {data_collection[selected_feature]['total_count']}',)
            st.write(f'Unexpected observations: {data_collection[selected_feature]['not_valid_count']}',)
            st.write(f'Unexpected observations in %: {data_collection[selected_feature]['not_valid_perc']}',)

        st.selectbox(
            label='Choose method to determine feature ranges:',
            options=['Interquartil-Range-Method', 'Standard-Deviation-Method', 'Modified-Z-Score-Method', 'Advanced-Gamma-Method'],
            key='method_selector_3',
                index=['Interquartil-Range-Method', 'Standard-Deviation-Method', 'Modified-Z-Score-Method', 'Advanced-Gamma-Method'].index(st.session_state.selected_method),
                on_change=update_method,
                args=('method_selector_3',) 
        ) 


        col1, col2, col3 = st.columns(3)
        with col1:
            st.page_link("pages/statistics.py", label="Data Statistics", icon="📊")
        with col2:
            st.page_link("pages/visualization.py", label="Data Visualization", icon="📈")
        with col3:
            st.page_link("app.py", label="Home", icon="🏠")

else: 
    st.write("Upload some data")
    st.page_link("app.py", label="Home", icon="🏠")

