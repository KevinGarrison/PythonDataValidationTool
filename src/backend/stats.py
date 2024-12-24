import pandas as pd, numpy as np
import streamlit as st
from dataclasses import dataclass
from scipy.stats import skew, kurtosis
import numpy as np
import plotly.express as px
import scipy.stats as stats
from backend.colors import ColorPalette

@dataclass
class Statistics:
    """A data analysis class that provides methods for summarizing data, detecting outliers, 
       and normalizing features using various statistical and machine learning approaches."""


    @st.cache_data
    def iqr_approach(self, df):                                                                                                 
        """Applies the IQR method to calculate feature bounds for outlier detection."""
        df = df.select_dtypes(include=['number'])
        summary = df.describe()

        return pd.DataFrame({
            'feature': summary.columns,
            'lower_bound': round(summary.loc['25%'] - 1.5 * (summary.loc['75%'] - summary.loc['25%']),2),
            'upper_bound': round(summary.loc['75%'] + 1.5 * (summary.loc['75%'] - summary.loc['25%']),2)
        })


    @st.cache_data                                                                                                          
    def std_approach(self, df, alpha: int = 3):
        """Applies a standard deviation method to define outlier thresholds."""
        df = df.select_dtypes(include=['number'])
        summary = df.describe()

        return pd.DataFrame({
            'feature': summary.columns,
            'lower_bound': round(summary.loc['mean'] - alpha * summary.loc['std'],2),
            'upper_bound': round(summary.loc['mean'] + alpha * summary.loc['std'],2)
        })
    

    @st.cache_data
    def modified_z_score_approach(self, df, k: float = 3.5):
        """Berechnet Modified Z-Scores und identifiziert Ausreißer."""
        df = df.select_dtypes(include=['number'])

        filter_ranges = []

        for column in df.columns:
            median = df[column].median()
            mad = np.median(np.abs(df[column] - median))
            filter_ranges.append({
                'feature': column,
                'lower_bound': round(median - k * mad / 0.6745,2),
                'upper_bound': round(median + k * mad / 0.6745,2)
            })
        
        return pd.DataFrame(filter_ranges)
           
    
    @st.cache_data
    def gamma_method_modified(self, df, alphas: int = 6, alphak: int = 30, beta_1=2, beta_2=2, gamma=2):
        """gets tresholds using skewness and kurtosis with flexible gamma adjustments."""
        df = df.select_dtypes(include=['number'])
        
        filter_ranges = []
        for column in df.columns:
            data = df[column]
            
            mean = data.mean()
            sigma = data.std()

            z_scores = (data - mean) / sigma

            skew_result, kurt_result = abs(skew(z_scores)), abs(kurtosis(z_scores))
            if skew_result < alphas and kurt_result < alphak:
                filter_ranges.append({
                'feature': column,
                'lower_bound': round(mean - beta_1 * sigma,2),
                'upper_bound': round(mean + beta_2 * sigma,2)
                })
            else:
                filter_ranges.append({
                    'feature': column,
                    'lower_bound': round(mean - gamma * beta_1 * sigma,2),
                    'upper_bound': round(mean + gamma * beta_2 * sigma,2)
                })
            
        return pd.DataFrame(filter_ranges)

    @st.cache_data
    def boxplot_px(self, data, ranges, column, bound_color='color_bounds_boxplot'):
        
        data = data
        np_array_data = np.array(data[column])  
        sorted_data = np.sort(np_array_data)

        df_ranges = ranges
        row = df_ranges[df_ranges['feature'] == column]

        lower_bound = row['lower_bound'].values[0]
        upper_bound = row['upper_bound'].values[0]

        box_data = pd.DataFrame({
            'Values': sorted_data,
            'Type': ['Normal'] * len(sorted_data)
        })

        fig = px.box(box_data, x='Values', color='Type', 
                    title=f'Boxplot for {column}',
                    points='all',
                    width=1400,
                    height=500)  
        
        fig.update_traces(
            marker=dict(color=ColorPalette.get_color_hex('datapoints_boxplot')),  
            line=dict(color=ColorPalette.get_color_hex('boxplot'))  
        )

        fig.add_vline(
            x=lower_bound,
            line_color=ColorPalette.get_color_hex(bound_color),
            line_width=2,
            line_dash="dash",
            annotation_text="Lower Bound",
            annotation_position="top right"
        )
        fig.add_vline(
            x=upper_bound,
            line_color=ColorPalette.get_color_hex(bound_color),
            line_width=2,
            line_dash="dash",
            annotation_text="Upper Bound",
            annotation_position="top right"
        )

        fig.update_layout(
            xaxis_title='Values',
            yaxis_title='',
            showlegend=False
        )

        st.plotly_chart(fig)



    # Example function to plot histogram with theoretical distributions
    def plot_histogram_with_theoretical(self, data, selected_feature, bins=30, dist_name='norm'):
    
        fig = px.histogram(
            data,
            x=selected_feature,
            nbins=bins,
            title=f'Histogram of {selected_feature} Distribution',
            color_discrete_sequence=[ColorPalette.get_color_hex('histogram')],
            marginal='box',
            width=1400,
            height=500
        )

        fig.update_traces(marker=dict(line=dict(color=ColorPalette.get_color_hex('datapoints_histogram'), width=1)))
        fig.update_layout(xaxis_title=selected_feature, yaxis_title='Frequency')

        np_array_data = np.array(data[selected_feature])
        x_values = np.linspace(min(np_array_data), max(np_array_data), 1000)

        if dist_name == 'norm':
            mean, std = np.mean(np_array_data), np.std(np_array_data)
            pdf = stats.norm.pdf(x_values, loc=mean, scale=std)
        elif dist_name == 'chisquare':
            df = len(np_array_data) - 1
            pdf = stats.chi2.pdf(x_values, df)
        elif dist_name == 'expon':
            scale = np.mean(np_array_data)
            pdf = stats.expon.pdf(x_values, scale=scale)
        elif dist_name == 'uniform':
            low, high = np.min(np_array_data), np.max(np_array_data)
            pdf = stats.uniform.pdf(x_values, loc=low, scale=high - low)
        elif dist_name == 't':
            df = len(np_array_data) - 1
            pdf = stats.t.pdf(x_values, df)
        elif dist_name == 'lognorm':
            shape, loc, scale = stats.lognorm.fit(np_array_data, floc=0)  
            pdf = stats.lognorm.pdf(x_values, s=shape, loc=loc, scale=scale)

        pdf = pdf * len(np_array_data) * (max(np_array_data) - min(np_array_data)) / bins

        fig.add_scatter(
            x=x_values,
            y=pdf,
            mode='lines',
            line=dict(color=ColorPalette.get_color_hex('datapoints_histogram'), width=2),
            name=f'{dist_name.capitalize()} PDF'
        )

        st.plotly_chart(fig)



    @st.cache_data
    def qq_plot(self, dist_name, data, sample_size, random_state=42):
        data = data

        np.random.seed(random_state)
        
        # Estimate parameters based on the sample data
        if dist_name == 'norm':
            # Estimate mean and std for normal distribution
            params = {'mean': np.mean(data), 'std': np.std(data)}
        elif dist_name == 'chisquare':
            # Estimate degrees of freedom for Chi-squared distribution
            params = {'df': np.mean(data) ** 2 / np.var(data)}
        elif dist_name == 'expon':
            # Estimate scale parameter for exponential distribution
            params = {'scale': np.mean(data)}
        elif dist_name == 'uniform':
            # Estimate low and high for uniform distribution
            params = {'low': np.min(data), 'high': np.max(data)}
        elif dist_name == 't':
            # Estimate degrees of freedom for t-distribution
            params = {'df': sample_size - 1}
        elif dist_name == 'lognorm':
            # Estimate shape, loc, and scale for log-normal distribution
            shape, loc, scale = stats.lognorm.fit(data, floc=0)  # Fix location to 0 for simpler estimation
            params = {'shape': shape, 'scale': scale}

        
        
        # Compute theoretical quantiles based on the estimated parameters
        if dist_name == 'norm':
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, sample_size), loc=params['mean'], scale=params['std'])
        elif dist_name == 'chisquare':
            theoretical_quantiles = stats.chi2.ppf(np.linspace(0.01, 0.99, sample_size), df=params['df'])
        elif dist_name == 'expon':
            theoretical_quantiles = stats.expon.ppf(np.linspace(0.01, 0.99, sample_size), scale=params['scale'])
        elif dist_name == 'uniform':
            theoretical_quantiles = stats.uniform.ppf(np.linspace(0.01, 0.99, sample_size), loc=params['low'], scale=params['high'] - params['low'])
        elif dist_name == 't':
            theoretical_quantiles = stats.t.ppf(np.linspace(0.01, 0.99, sample_size), df=params['df'])
        elif dist_name == 'lognorm':
            theoretical_quantiles = stats.lognorm.ppf(np.linspace(0.01, 0.99, sample_size), s=params['shape'], scale=params['scale'])

        
        # Sort the sample values
        sorted_samples = np.sort(data)
        
        # Create a DataFrame for plotting
        df = pd.DataFrame({
            'Theoretical Quantiles': theoretical_quantiles,
            'Sample Quantiles': sorted_samples
        })
        
        fig = px.scatter(df, x='Theoretical Quantiles', y='Sample Quantiles',
                 title=f'QQ Plot for {dist_name.capitalize()} Distribution',
                 labels={'Theoretical Quantiles': 'Theoretical Quantiles',
                         'Sample Quantiles': 'Sample Quantiles'},
                 color_discrete_sequence=[ColorPalette.get_color_hex('qq_plot')])  # Set color to blue

        
        # Add a line for perfect agreement
        fig.add_scatter(x=theoretical_quantiles, y=theoretical_quantiles,
                        mode='lines', name='45-degree line', line=dict(dash='dash', color=ColorPalette.get_color_hex('degree_45_line')))
        
        fig.update_layout(
            width=1400,  # Set the width of the plot
            height=500,  # Set the height of the plot
        )
        # Show the plot
        st.plotly_chart(fig)





