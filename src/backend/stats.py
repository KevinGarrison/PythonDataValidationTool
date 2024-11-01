import pandas as pd, numpy as np
import streamlit as st
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from scipy.stats import skew, kurtosis
import numpy as np
import plotly.express as px

@dataclass
class Statistics:
    """A data analysis class that provides methods for summarizing data, detecting outliers, 
       and normalizing features using various statistical and machine learning approaches."""


    @st.cache_data
    def feature_stats_summarize(self):
        """Summarizes statistical features of the dataset."""
        try:
            df = st.session_state.data_numerical
            if df is None or df.empty:
                st.warning("The data is empty. Please load data first.")
            st.session_state.stats_summarize = df.describe()
        except Exception as e:
            st.error(f"An error occurred while summarizing statistics: {e}")
    

    @st.cache_data
    def normalize_numerical_data(self, df):
        """Standardizes numerical data using Z-score normalization."""
        try:
            df = df
            columns = df.columns
            scaler = StandardScaler()
            standardized_data = scaler.fit_transform(df)
            st.session_state.scaler_mean = scaler.mean_
            st.session_state.scaler_scale = scaler.scale_
            return pd.DataFrame(standardized_data, columns=columns)
        except Exception as e:
            st.error(f"An error occurred during normalization: {e}")
            return None
    

    @st.cache_data
    def inverse_normalize_data(self):
        try:    
            if 'scaler_mean' in st.session_state and 'scaler_scale' in st.session_state:
                df = st.session_state.data_filtered
                columns = df.columns                                                                  
                scaler = StandardScaler()
                scaler.mean_ = st.session_state.scaler_mean
                scaler.scale_ = st.session_state.scaler_scale
                original_data = scaler.inverse_transform(df)           
                st.session_state.data_final = pd.DataFrame(original_data, columns=columns)
        except Exception as e:
                st.error(f"An error occurred during inverse normalization: {e}")
    

    @st.cache_data
    def inverse_filter_ranges(self, filter_ranges):
        try:
            ranges = filter_ranges 
            mean = st.session_state.scaler_mean
            std = st.session_state.scaler_scale
            ranges['lower_bound'] = round(ranges['lower_bound'] * std + mean, 2)
            ranges['upper_bound'] = round(ranges['upper_bound'] * std + mean, 2)
            return ranges
        except Exception as e:
            st.error(f"An error occurred during inverse normalization: {e}")
            return None
        

    @st.cache_data
    def iqr_approach(self, df):                                                                                                 
        """Applies the IQR method to calculate feature bounds for outlier detection."""
        df = df 
        summary = df.describe()
        return pd.DataFrame({
            'feature': summary.columns,
            'lower_bound': round(summary.loc['25%'] - 1.5 * (summary.loc['75%'] - summary.loc['25%']),2),
            'upper_bound': round(summary.loc['75%'] + 1.5 * (summary.loc['75%'] - summary.loc['25%']),2)
        })


    @st.cache_data                                                                                                          
    def z_score_approach(self, df, alpha: int = 3):
        """Applies a standard deviation method to define outlier thresholds."""
        df = df
        summary = df.describe()

        return pd.DataFrame({
            'feature': summary.columns,
            'lower_bound': round(summary.loc['mean'] - alpha * summary.loc['std'],2),
            'upper_bound': round(summary.loc['mean'] + alpha * summary.loc['std'],2)
        })
    

    @st.cache_data
    def modified_z_score_approach(self, df, alpha: int = 3):
        """Calculates modified Z-scores and defines outlier thresholds."""
        filter_ranges = []

        for column in df.select_dtypes(include=[np.number]).columns:
            median = df[column].median()
            
            absolute_deviations = np.abs(df[column] - median)
            mad = absolute_deviations.median()
       
            if mad == 0:
                lower_bound = upper_bound = median 
            else:
                lower_bound = round(median - alpha * mad, 2)
                upper_bound = round(median + alpha * mad, 2)

            # Append the results for the column
            filter_ranges.append({
                'feature': column,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            })
        
        return pd.DataFrame(filter_ranges)      


    @st.cache_data
    def f1(self, z: pd.Series, beta_1: int, beta_2: int) -> pd.Series:
        """Detects outliers based on specified z-score thresholds."""
        anomaly_mask = ~((-beta_1 < z) & (z < beta_2))
        return anomaly_mask


    @st.cache_data
    def f2(self, z: pd.Series, beta_1: int, beta_2: int, gamma: int = 3) -> pd.Series:
        """Combines z-score and isolation forest methods for outlier detection."""
        mask_f1 = self.f1(z, gamma * beta_1, gamma * beta_2)
        mask_iforest = self.isolation_forest(z)
        return mask_f1 & mask_iforest


    @st.cache_data
    def gamma_outlier(self, df, alphas: int = 6, alphak: int = 30):
        """Detects outliers using skewness and kurtosis thresholds with flexible gamma adjustments."""
        df = df
        if df is None:
            st.warning("Standardized data not found. Please normalize data first.")
        filter_ranges = []
        for column in df.columns:
            data = df[column]
            skew_result, kurt_result = abs(skew(data)), abs(kurtosis(data))
            if skew_result < alphas and kurt_result < alphak:
                mask = self.f1(data, beta_1=skew_result, beta_2=kurt_result)
            else:
                mask = self.f2(data, beta_1=skew_result, beta_2=kurt_result, gamma=3)
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
    def boxplot_px(self, data, ranges, column, bound_color='yellow'):
        # Prepare the data
        np_array_data = np.array(data[column])  
        sorted_data = np.sort(np_array_data)

        df_ranges = ranges
        row = df_ranges[df_ranges['feature'] == column]

        lower_bound = row['lower_bound'].values[0]
        upper_bound = row['upper_bound'].values[0]

        # Create a DataFrame for Plotly
        box_data = pd.DataFrame({
            'Values': sorted_data,
            'Type': ['Normal'] * len(sorted_data)
        })

        # Create the boxplot
        fig = px.box(box_data, x='Values', color='Type', 
                    title=f'Boxplot for {column}',
                    points='all',
                    width=1400,
                    height=500)  
        
        fig.update_traces(
        marker=dict(color='green'),  
        line=dict(color='white')     
        )

        # Add vertical lines for median and bounds
        fig.add_vline(x=lower_bound, line_color=bound_color, line_width=2, line_dash="dash", annotation_text="Lower Bound", annotation_position="top right")
        fig.add_vline(x=upper_bound, line_color=bound_color, line_width=2, line_dash="dash", annotation_text="Upper Bound", annotation_position="top right")

        # Update layout for aesthetics
        fig.update_layout(
            xaxis_title='Values',
            yaxis_title='',
            showlegend=False
        )

        st.plotly_chart(fig)



