import streamlit as st
from backend.utils import Utilitis

utils = Utilitis()

st.set_page_config(initial_sidebar_state='collapsed')

# Callback function
def set_page(page):
    if page == 'upload':
        st.session_state.algo_runned = False    
    st.session_state.page = page

def update_method(key, *args):
    st.session_state.selected_method = st.session_state[key]

if 'process_data' in st.session_state:
    del st.session_state.process_data

if 'page' not in st.session_state:
    # Page state
    st.session_state['page'] = 'upload'
    # Data states
    st.session_state['data'] = None
    st.session_state['data_numerical'] = None
    st.session_state['data_filtered'] = None
    st.session_state['data_final'] = None
    # Filter range state
    st.session_state['filter_ranges'] = None
    # Action states
    st.session_state['data_cleaned'] = False
    st.session_state['upload_new_data'] = False
    # Keys for dropdown menus
    st.session_state['selected_method'] = 'Interquartil-Range-Method'
    st.session_state['selected_feature'] = None
    # Great Expectation states
    st.session_state['batch_definition'] = None
    st.session_state['batch_parameters'] = None
    st.session_state['min_max_exp'] = None

if st.session_state.page == 'upload':

    st.header("Numerical Data Validation Tool", divider="rainbow")
    st.write("Upload your data below:",)
    uploaded_file = st.file_uploader(label="Choose a file", type=['csv', 'xls', 'xlsx']) 
    if uploaded_file is not None:
        df = utils.load_data(uploaded_file=uploaded_file)
        st.session_state.data = df
        st.write("Dataset:",)
        st.write(df,)
        st.selectbox(
        label='Choose method to determine feature ranges (only for numerical values):',
        options=['Interquartil-Range-Method', 'Standard-Deviation-Method', 'Modified-Z-Score-Method', 'Advanced-Gamma-Method'],
        key='method_selector_0',  
        index=['Interquartil-Range-Method', 'Standard-Deviation-Method', 'Modified-Z-Score-Method', 'Advanced-Gamma-Method'].index(st.session_state.selected_method),
        on_change=update_method,
        args=('method_selector_0',) 
        ) 
        st.page_link("pages/download.py", label="Determin feature ranges", icon="📐") 
        col1, col2 = st.columns(2)
        with col1:
            st.page_link("pages/statistics.py", label="Data Statistics", icon="📊")
        with col2:
            st.page_link("pages/visualization.py", label="Data Visualization", icon="📈")
        
        st.button("Upload new data", on_click=set_page, args=['upload'])
            


    

         