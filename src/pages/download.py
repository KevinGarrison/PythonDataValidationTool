import streamlit as st
import pandas as pd, numpy as np
from backend.utils import Utilitis
from backend.stats import Statistics
from app import update_method


utils = Utilitis()
stats = Statistics()

st.header("Recommended Feature Ranges", divider="rainbow")

data = st.session_state.data

with st.spinner("Processing... Please wait."):
    method = st.session_state.selected_method
    original_ranges = utils.run_algorithm(data, method)
    utils.setup_gx()
    batch_parameters = {"dataframe": st.session_state.data}
    batch = st.session_state.batch_definition.get_batch(
        batch_parameters=batch_parameters
    )
    st.write('Results Overview:', divider="rainbow")
    st.write(original_ranges, divider="rainbow")
    st.session_state.data_collection = dict()

    for index, row in original_ranges.iterrows():
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
        data_collection.append(round(validation_result['result']['unexpected_percent'], 3))
        data_collection.append(validation_result['result']['partial_unexpected_list'])
        data_collection.append(validation_result['result']['partial_unexpected_counts'])  # list of dicts
        data_collection.append(validation_result['result']['partial_unexpected_index_list'])
        st.session_state.data_collection[feature] = data_collection
        #st.write(validation_result)
    #time.sleep(2)
        

selected_feature = st.selectbox(label="Choose a feature to get expected value range and visualisation:",options=list(original_ranges['feature']))

df_num = st.session_state.data

if selected_feature in list(df_num.columns):
    if st.session_state.data_collection[selected_feature][1] == True:
        st.write(f'Data range for **{selected_feature}** meets expectations with approach:', divider="rainbow")
        st.write(f"{st.session_state.selected_method}", divider="rainbow")
    else:
        st.write(f'Data range for **{selected_feature}** is not as expected with approach:', divider="rainbow")
        st.write(f"{st.session_state.selected_method}", divider="rainbow")
    stats.boxplot_px(df_num, original_ranges, selected_feature)
    
    for i, data in enumerate(st.session_state.data_collection[selected_feature]):
        match i:
            case 0:
                lower = data['lower_bound'][0]
                upper = data['upper_bound'][0]
                st.write(f'Expected lower bound: {lower}', divider="rainbow")
                st.write(f'Expected upper bound: {upper}', divider="rainbow")
            case 2: 
                st.write(f'Observations total: {data}', divider="rainbow")
            case 3:
                st.write(f'Unexpected observations: {data}', divider="rainbow")
            case 4:
                st.write(f'Unexpected observations in %: {data}', divider="rainbow")
            case 5:
                if data:
                    distinct_data = set(data)
                    formatted_text_distinct =  ', '.join(map(str, distinct_data)) 
                    st.write(f'Unexpected values as distinct list: [{formatted_text_distinct}]', divider="rainbow")
            case 6:
                if data: 
                    df = pd.DataFrame(data)
                    sorted_df = df.sort_values(by='value')
                    st.write(sorted_df, divider="rainbow")

st.selectbox(
    label='Choose method to determine feature ranges:',
    options=['Interquartil-Range-Method', 'Standard-Deviation-Method', 'Modified-Z-Score-Method', 'Advanced-Gamma-Method'],
    key='method_selector_3',
        index=['Interquartil-Range-Method', 'Standard-Deviation-Method', 'Modified-Z-Score-Method', 'Advanced-Gamma-Method'].index(st.session_state.selected_method),
        on_change=update_method,
        args=('method_selector_3',) 
) 


#st.page_link("pages/download.py", label="Determin feature ranges", icon="📐")

col1, col2, col3 = st.columns(3)
with col1:
    st.page_link("pages/statistics.py", label="Data Statistics", icon="📊")
with col2:
    st.page_link("pages/visualization.py", label="Data Visualization", icon="📈")
with col3:
    st.page_link("app.py", label="Home", icon="🏠")

