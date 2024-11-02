import pandas as pd, numpy as np
import streamlit as st
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
from scipy.stats import skew, kurtosis
import numpy as np
import plotly.express as px

@dataclass
class Statistics:
    """A data analysis class that provides methods for summarizing data, detecting outliers, 
       and normalizing features using various statistical and machine learning approaches."""


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
    def std_approach(self, df, alpha: int = 3):
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
        df = df
        filter_ranges = []

        for column in df.columns:
            median = df[column].median()
            
            absolute_deviations = np.abs(df[column] - median)
            mad = absolute_deviations.median()
       
            if mad == 0:
                lower_bound = upper_bound = median 
            else:
                lower_bound = round(median - alpha * mad, 2)
                upper_bound = round(median + alpha * mad, 2)

            
            filter_ranges.append({
                'feature': column,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            })
        
        return pd.DataFrame(filter_ranges)      


    @st.cache_data
    def f1(self, z: pd.Series, beta_1: int, beta_2: int) -> pd.Series:
        """Detects outliers based on specified z-score thresholds."""
        z = z
        anomaly_mask = (z <= (-beta_1)) | (z >= beta_2)

        return anomaly_mask


    @st.cache_data
    def f2(self, z: pd.Series, beta_1: int, beta_2: int, gamma: int = 3) -> pd.Series:
        """Combines z-score and isolation forest methods for outlier detection."""
        z = z
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
            
            mean = data.mean()
            std_dev = data.std()

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
    def boxplot_px(self, data, ranges, column, bound_color='yellow'):
        
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



