import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Streamlit Header
st.header("Visualization of your data")

# Assuming 'data' is already set in session state
data = st.session_state.data

selections =  ["All data"] + list(data.columns)
# Dropdown to select the feature to plot
feature = st.selectbox('Select a specific feature to visualize:', selections)



if feature == 'All data':
    st.write(data)
elif feature in list(data.columns):
    # Set a common figure size
    fig_width = 14
    fig_height = 8
    
    col1, col2 = st.columns([1.5,4])
    with col1:
        st.write(data[feature])
    with col2:
        #Boxplot
        plt.figure(figsize=(4, 1))  
        sns.boxplot(x=data[feature], color='lightblue')
        plt.xlabel(feature)
        plt.title(f'Box Plot of {feature}')
        st.pyplot(plt)
        
    # Slider to select the number of bins for the histogram
    bins = st.slider('Select number of bins for the histogram:', min_value=5, max_value=20, value=10)

    # Histogram
    plt.figure(figsize=(14, 8))
    sns.histplot(data[feature], bins=bins, kde=True, color='blue', edgecolor='black')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {feature} Distribution with {bins} bins')
    st.pyplot(plt)
    

st.page_link("pages/statistics.py", label="Data Statistics", icon="📊")
st.page_link("app.py", label="Home", icon="🏠")









# Optionally, show the data as well

