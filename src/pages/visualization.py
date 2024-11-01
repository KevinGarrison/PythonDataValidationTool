import streamlit as st
import plotly.express as px


st.header("Visualization of your data") # TODO of your data without anomalies

data = st.session_state.data_final

selections =  ["-Select feature-"] + list(data.columns)

feature = st.selectbox('Select a specific feature to visualize:', selections)

# TODO Plots via a selection

if feature in list(data.columns):
    # Slider to select the number of bins for the histogram
    bins = st.slider('Select number of bins for the histogram:', min_value=5, max_value=20, value=10) # TODO Define the range of the bins

    
    # Create histogram with KDE
    fig = px.histogram(data, x=feature, nbins=bins, 
                    title=f'Histogram of {feature} Distribution with {bins} bins',
                    color_discrete_sequence=['white'],
                    marginal='box',
                    width=1400,
                    height=500
                    )  

    fig.update_traces(marker=dict(line=dict(color='black', width=1)))  # Edge color for bars
    fig.update_layout(xaxis_title=feature, yaxis_title='Frequency')

    st.plotly_chart(fig)
    selections_plot = ['Box Plot', 'Violin Plot', 'Density Plot']
    plot = st.selectbox('Select a Chart', selections_plot)

    if plot == 'Box Plot':
        fig_chart = px.box(data, y=feature, title=f'Box Plot of {feature}')
    elif plot == 'Violin Plot':
        fig_chart = px.violin(data, y=feature, title=f'Violin Plot of {feature}')
    elif plot == 'Density Plot':
        fig_chart = px.density_contour(data, x=feature, title=f'Density Plot of {feature}')
    if fig_chart is not None:
        st.plotly_chart(fig_chart)
    
method = st.selectbox(label='Choose method to determine feature ranges:',options=['Interquartil-Range-Method', 'Z-Score-Method', 'Modified-Z-Score-Method', 'Advanced-Gamma-Method'])
st.session_state.method = method
st.page_link("pages/download.py", label="Determin feature ranges", icon="📐") 

col1, col2 = st.columns(2)
with col1:
    st.page_link("pages/statistics.py", label="Data Statistics", icon="📊")
with col2:
    st.page_link("app.py", label="Home", icon="🏠")
