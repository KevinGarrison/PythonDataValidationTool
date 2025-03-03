import streamlit as st
from backend.utils import Utilitis
import pandas as pd

utils = Utilitis()

st.set_page_config(            
    initial_sidebar_state="collapsed"  
)


# Callback function
def set_page(page):
    if page == 'upload':
        st.session_state.algo_runned = False    
    st.session_state.page = page

def update_method(key, *args):
    st.session_state.selected_method = st.session_state[key]

if 'init' not in st.session_state:
    st.session_state['init'] = True
    # Data states
    st.session_state['data'] = pd.DataFrame()
    # Filter range state
    st.session_state['filter_ranges'] = None
    # Action states
    st.session_state['data_cleaned'] = False
    st.session_state['uploaded_data'] = False
    # Keys for dropdown menus
    st.session_state['selected_method'] = 'Interquartil-Range-Method'
    st.session_state['selected_feature'] = None


st.header("Numerical Data Validation Tool", divider="rainbow")
st.write("Upload your data below:",)
uploaded_file = st.file_uploader(label="Choose a file", type=['csv', 'xls', 'xlsx']) 

if uploaded_file is not None:
    df = utils.load_data(uploaded_file=uploaded_file)
    st.session_state.data = df
    st.write("Dataset:",)
    st.write(df,)
    if not st.session_state.data.empty:
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
    
     


    

         