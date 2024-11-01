import streamlit as st

st.header("Statistics of your data")

df = st.session_state.data_final

selections =  ["All data"] + list(df.columns)
feature = st.selectbox('Select a feature to get the statistics:', selections)

if feature == "All data":
    description = df.describe()
    st.write(description)
if feature in list(df.columns):
    description = df.describe() 
    st.write(description[feature])


st.session_state.method = None
method = st.selectbox(label='Choose method to determine feature ranges:',options=['Interquartil-Range-Method', 'Z-Score-Method', 'Advanced-Gamma-Method'])
st.session_state.method = method
st.page_link("pages/download.py", label="Determin feature ranges", icon="📐") 

col1, col2 = st.columns(2)
with col1:
    st.page_link("pages/visualization.py", label="Data Visualization", icon="📈")
with col2:
    st.page_link("app.py", label="Home", icon="🏠")

