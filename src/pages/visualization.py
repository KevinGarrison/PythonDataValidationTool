import streamlit as st
import plotly.express as px
from backend import stats
from app import update_method
from backend.colors import ColorPalette

stats_obj = stats.Statistics()

st.header("Visualization of your data", divider="rainbow") 

data = st.session_state.data

st.write(data)

if not data.empty:

    data = data.select_dtypes(include=['number'])

    selections =  ["-Select feature-"] + list(data.columns)

    selected_feature = st.selectbox('Select a specific feature to visualize:', selections)

    if selected_feature in list(data.columns):
        
        
        selected_dist = None
        activate = st.toggle('Expert View')
        if activate:
            distributions = ['norm', 'chisquare', 'expon', 'uniform', 'binomial', 'lognorm', 'pareto']
            selected_dist = st.selectbox('Select a distribution for the histogram:', distributions)
            
        bins = st.slider('Select number of bins for the histogram:', min_value=5, max_value=30, value=5) 
        if selected_dist == 'lognorm' and (data[selected_feature].isnull().any() or (data[selected_feature] <= 0).any()):
                st.error("Maximum likelihood estimation with lognormal distribution requires that 0.0 < (x - loc)/scale < inf for each x in data")
        elif not activate:
            stats_obj.plot_histogram_with_theoretical(data=data, selected_feature=selected_feature, bins=bins, dist_name='norm', key=selected_feature)
        else:
            stats_obj.qq_plot(dist_name=selected_dist, data=data, sample_size=len(data))
            stats_obj.plot_histogram_with_theoretical(data=data, selected_feature=selected_feature, bins=bins, dist_name=selected_dist, key=selected_feature)

        selections_plot = ['Box Plot', 'Violin Plot']
        plot = st.selectbox('Select a Chart', selections_plot)

        if plot == 'Box Plot':
            fig_chart = px.box(data, y=selected_feature, title=f'Box Plot of {selected_feature}')
            fig_chart.update_traces(marker=dict(color=ColorPalette.get_color_hex('boxplot_2')))  
        elif plot == 'Violin Plot':
            fig_chart = px.violin(data, y=selected_feature, title=f'Violin Plot of {selected_feature}')
            fig_chart.update_traces(marker=dict(color=ColorPalette.get_color_hex('violinplot')))
        if fig_chart is not None:
            st.plotly_chart(fig_chart, key='plot2')

        
    st.selectbox(
        label='Choose method to determine feature ranges:',
        options=['Interquartil-Range-Method', 'Standard-Deviation-Method', 'Modified-Z-Score-Method', 'Advanced-Gamma-Method'],
        key='method_selector_1',  
        index=['Interquartil-Range-Method', 'Standard-Deviation-Method', 'Modified-Z-Score-Method', 'Advanced-Gamma-Method'].index(st.session_state.selected_method),
        on_change=update_method,
        args=('method_selector_1',) 
    ) 
    st.page_link("pages/download.py", label="Determin feature ranges", icon="📐") 

    col1, col2 = st.columns(2)
    with col1:
        st.page_link("pages/statistics.py", label="Data Statistics", icon="📊")
    with col2:
        st.page_link("app.py", label="Home", icon="🏠")

else: 
    st.write("Upload some data")
    st.page_link("app.py", label="Home", icon="🏠")
