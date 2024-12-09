import pandas as pd, numpy as np
import streamlit as st
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
from scipy.stats import skew, kurtosis
import numpy as np
import plotly.express as px
import scipy.stats as stats

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
    def f1(self, z: pd.Series, beta_1: int, beta_2: int) -> pd.Series:
        """Detects outliers based on specified z-score thresholds."""
        z = z
        anomaly_mask = (z <= (-beta_1)) | (z >= beta_2)

        return anomaly_mask


    @st.cache_data
    def f2(self, z: pd.Series, beta_1: int, beta_2: int, gamma: int = 2) -> pd.Series:
        """Combines z-score and isolation forest methods for outlier detection."""
        z = z
        mask_f1 = self.f1(z, gamma * beta_1, gamma * beta_2)
        mask_iforest = self.isolation_forest(z)

        return mask_f1 & mask_iforest


    @st.cache_data
    def gamma_outlier(self, df, alphas: int = 6, alphak: int = 30):
        """Detects outliers using skewness and kurtosis thresholds with flexible gamma adjustments."""
        df = df
        
        filter_ranges = []
        for column in df.columns:
            data = df[column]
            
            mean = data.mean()
            std_dev = np.sqrt(2 * 1)

            z_scores = (data - mean) / std_dev

            skew_result, kurt_result = abs(skew(z_scores)), abs(kurtosis(z_scores))
            if skew_result < alphas and kurt_result < alphak:
                mask = self.f1(z_scores, beta_1=3, beta_2=3)
            else:
                mask = self.f2(z_scores, beta_1=3, beta_2=3, gamma=2)
            filtered_data = data[~mask]
            filter_ranges.append({
                'feature': column,
                'lower_bound': round(filtered_data.min(),2),
                'upper_bound': round(filtered_data.max(),2)
            })
            
        return pd.DataFrame(filter_ranges)


    @st.cache_data
    def isolation_forest(self, z: pd.Series) -> pd.Series:
        """Detects outliers using the Isolation Forest algorithm."""
        iso_forest = IsolationForest(random_state=42)

        return pd.Series(iso_forest.fit_predict(z.values.reshape(-1, 1)), index=z.index) == 1
    
    @st.cache_data
    def gamma_method_modified(self, df, alphas: int = 6, alphak: int = 30, beta_1=2, beta_2=2, gamma=2, k=1):
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
                st.write('Opt1')
                filter_ranges.append({
                'feature': column,
                'lower_bound': round(mean - beta_1 * sigma,2),
                'upper_bound': round(mean + beta_2 * sigma,2)
                })
            else:
                st.write('Opt2')    
                filter_ranges.append({
                    'feature': column,
                    'lower_bound': round(mean - gamma * beta_1 * sigma,2),
                    'upper_bound': round(mean + gamma * beta_2 * sigma,2)
                })
            
        return pd.DataFrame(filter_ranges)

    @st.cache_data
    def boxplot_px(self, data, ranges, column, bound_color='yellow'):
        
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
        marker=dict(color='green'),  
        line=dict(color='white')     
        )

       
        fig.add_vline(x=lower_bound, line_color=bound_color, line_width=2, line_dash="dash", annotation_text="Lower Bound", annotation_position="top right")
        fig.add_vline(x=upper_bound, line_color=bound_color, line_width=2, line_dash="dash", annotation_text="Upper Bound", annotation_position="top right")

       
        fig.update_layout(
            xaxis_title='Values',
            yaxis_title='',
            showlegend=False
        )

        st.plotly_chart(fig)


    @st.cache_data
    def qq_plot(self, dist_name, data, sample_size,  random_state=42):
        data = data

        np.random.seed(random_state)
        
        # Estimate parameters based on the sample data
        if dist_name == 'norm':
            # Estimate mean and std for normal distribution
            params = {'mean': np.mean(data), 'std': np.std(data)}
        elif dist_name == 'chisquare':
            # Estimate degrees of freedom for Chi-squared distribution
            # Use the formula for the degrees of freedom estimate: df = (n * sample variance) / mean
            params = {'df': np.mean(data) ** 2 / np.var(data)}
        elif dist_name == 'expon':
            # Estimate scale parameter for exponential distribution
            params = {'scale': np.mean(data)}
        elif dist_name == 'uniform':
            # Estimate low and high for uniform distribution
            params = {'low': np.min(data), 'high': np.max(data)}
        elif dist_name == 't':
            # Estimate degrees of freedom for t-distribution
            # This is a rough estimate for a sample from a t-distribution
            params = {'df': sample_size - 1}
        
        st.write(f"Estimated parameters for {dist_name.capitalize()} distribution: {params}")
        
        # Compute theoretical quantiles based on the estimated parameters
        if dist_name == 'norm':
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, sample_size), loc=params['mean'], scale=params['std'])
        elif dist_name == 'chisquare':
            theoretical_quantiles = stats.chi2.ppf(np.linspace(0.01, 0.99, sample_size), df=params['df'])
        elif dist_name == 'expon':
            theoretical_quantiles = stats.expon.ppf(np.linspace(0.01, 0.99, sample_size), scale=params['scale'])
        elif dist_name == 'uniform':
            theoretical_quantiles = stats.uniform.ppf(np.linspace(0.01, 0.99, sample_size), loc=params['low'], scale=params['high']-params['low'])
        elif dist_name == 't':
            theoretical_quantiles = stats.t.ppf(np.linspace(0.01, 0.99, sample_size), df=params['df'])
        
        # Sort the sample values
        sorted_samples = np.sort(data)
        
        # Create a DataFrame for plotting
        df = pd.DataFrame({
            'Theoretical Quantiles': theoretical_quantiles,
            'Sample Quantiles': sorted_samples
        })
        
        # Create the QQ plot using Plotly Express
        fig = px.scatter(df, x='Theoretical Quantiles', y='Sample Quantiles',
                        title=f'QQ Plot for {dist_name.capitalize()} Distribution',
                        labels={'Theoretical Quantiles': 'Theoretical Quantiles',
                                'Sample Quantiles': 'Sample Quantiles'})
        
        # Add a line for perfect agreement
        fig.add_scatter(x=theoretical_quantiles, y=theoretical_quantiles,
                        mode='lines', name='45-degree line', line=dict(dash='dash', color='red'))
        
        fig.update_layout(
        width=1400,  # Set the width of the plot
        height=500,  # Set the height of the plot
        )
        # Show the plot
        st.plotly_chart(fig)




