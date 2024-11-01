import streamlit as st
import pandas as pd, numpy as np
from backend.utils import Utilitis
from backend.stats import Statistics
import time

utils = Utilitis()
stats = Statistics()

st.header("Recommended Feature Ranges")

data = st.session_state.data_numerical

with st.spinner("Processing... Please wait."):
    method = st.session_state.method
    original_ranges = utils.run_algorithm(data, method)
    utils.setup_gx()
    batch_parameters = {"dataframe": st.session_state.data}
    batch = st.session_state.batch_definition.get_batch(
        batch_parameters=batch_parameters
    )        

    st.session_state.data_collection = dict()

    for _, row in original_ranges.iterrows():
        data_collection = list()
        feature = row['feature']
        lower = row['lower_bound']
        upper = row['upper_bound']
        datapoint = pd.DataFrame({'feature': [feature], 'lower_bound': [lower], 'upper_bound': [upper]})
        expectation_min_max = utils.define_column_values_between_exp(column=feature, min=lower, max=upper)
        validation_result = batch.validate(expectation_min_max)
        data_collection.append(datapoint)
        data_collection.append(validation_result['success'])
        data_collection.append(validation_result['result']['element_count'])
        data_collection.append(validation_result['result']['unexpected_count'])
        data_collection.append(round(validation_result['result']['unexpected_percent']))
        data_collection.append(validation_result['result']['partial_unexpected_list'])
        data_collection.append(validation_result['result']['partial_unexpected_counts']) # list of dicts
        data_collection.append(validation_result['result']['partial_unexpected_index_list'])
        st.session_state.data_collection[feature] = data_collection
    time.sleep(2)

feature = st.selectbox(label="Choose a feature to get expected value range and visualisation:",options=["-Select feature-"] + list(original_ranges['feature']))

df_num = st.session_state.data_numerical

if feature in list(df_num.columns):
    if st.session_state.data_collection[feature][1] == True:
        st.subheader('Data range meets expectations with approach:')
        st.markdown(f"<h2 style='color: yellow;'>{st.session_state.method}</h2>", unsafe_allow_html=True)
    else:
        st.subheader('Data range is not as expected with approach:')
        st.markdown(f"<h2 style='color: yellow;'>{st.session_state.method}</h2>", unsafe_allow_html=True)
    stats.boxplot_px(df_num, original_ranges, feature)

    for i, data in enumerate(st.session_state.data_collection[feature]):
        match i:
            case 0:
                lower = data['lower_bound'][0]
                upper = data['upper_bound'][0]
                st.markdown(f'Expected lower bound: <span style="color: yellow;">{lower}</span>', unsafe_allow_html=True)
                st.markdown(f'Expected upper bound: <span style="color: yellow;">{upper}</span>', unsafe_allow_html=True)
            case 2: 
                st.write('Observations total:', data)

            case 3:
                st.write('Unexpected observations:', data)

            case 4:
                if data > 0: 
                    st.write('Unexpected observations in %:', data)

            case 5:
                if data:
                    distinct_data = set(data)
                    formatted_text_distinct =  ', '.join(map(str, distinct_data)) 
                    st.write('Unexpected values as distinct list: [', formatted_text_distinct, ']')
            case 6:
                if data: 
                    df = pd.DataFrame(data)
                    sorted_df = df.sort_values(by='value')
                    st.write(sorted_df)


method = st.selectbox(label='Choose method to determine feature ranges:',options=['Interquartil-Range-Method', 'Z-Score-Method', 'Modified-Z-Score-Method', 'Advanced-Gamma-Method'])
st.session_state.method = None
st.session_state.method = method
st.page_link("pages/statistics.py", label="Data Statistics", icon="📊")
st.page_link("pages/visualization.py", label="Data Visualization", icon="📈") # TODO implement download function
st.page_link("pages/download.py", label="Determin feature ranges", icon="📐")
st.page_link("app.py", label="Home", icon="🏠")