import streamlit as st

st.header("Statistics of your data")
col1, col2 = st.columns(2)
with col1: 
    st.write(st.session_state.data)
with col2:
    st.write(st.session_state.data)
st.page_link("./app.py", label="Home", icon="🏠", )