import streamlit as st
from app import update_method

st.header("Statistics of your data")

df = st.session_state.data_final

selections =  ["All data"] + list(df.columns)
selected_feature = st.selectbox('Select a feature to get the statistics:', selections)

if selected_feature == "All data":
    description = df.describe()
    st.write(description)
if selected_feature in list(df.columns):
    description = df.describe() 
    st.write(description[selected_feature])



st.selectbox(
    label='Choose method to determine feature ranges:',
    options=['Interquartil-Range-Method', 'STD-Method', 'Modified-Z-Score-Method', 'Advanced-Gamma-Method'],
    key='method_selector_2',  
    index=['Interquartil-Range-Method', 'STD-Method', 'Modified-Z-Score-Method', 'Advanced-Gamma-Method'].index(st.session_state.selected_method),
    on_change=update_method,
    args=('method_selector_2',)  
)
st.page_link("pages/download.py", label="Determin feature ranges", icon="📐") 

col1, col2 = st.columns(2)
with col1:
    st.page_link("pages/visualization.py", label="Data Visualization", icon="📈")
with col2:
    st.page_link("app.py", label="Home", icon="🏠")

