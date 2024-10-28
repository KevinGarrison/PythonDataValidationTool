import pandas as pd, numpy as np
import streamlit as st
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from scipy.stats import skew, kurtosis
import numpy as np
import matplotlib.pyplot as plt

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
    def normalize_numerical_data(self):
        """Standardizes numerical data using Z-score normalization."""
        try:
            df = st.session_state.data_numerical
            columns = df.columns
            scaler = StandardScaler()
            standardized_data = scaler.fit_transform(df)
            st.session_state.scaler_mean = scaler.mean_
            st.session_state.scaler_scale = scaler.scale_
            st.session_state.data_standardized = pd.DataFrame(standardized_data, columns=columns)
        except Exception as e:
            st.error(f"An error occurred during normalization: {e}")
    
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
    def inverse_filter_ranges(self):
        try:
            ranges = st.session_state.filter_ranges.copy()  
            st.session_state.filter_ranges_original = ranges.copy()
            mean = st.session_state.scaler_mean
            std = st.session_state.scaler_scale
            ranges['lower_bound'] = round(ranges['lower_bound'] * std + mean, 2)
            ranges['upper_bound'] = round(ranges['upper_bound'] * std + mean, 2)
            st.session_state.filter_ranges_original = ranges
        except Exception as e:
            st.error(f"An error occurred during inverse normalization: {e}")
        

    @st.cache_data
    def iqr_approach(self):                                                                                                 
        """Applies the IQR method to calculate feature bounds for outlier detection."""
        df = st.session_state.data_standardized 
        summary = df.describe()
        st.session_state.filter_ranges = pd.DataFrame({
            'feature': summary.columns,
            'lower_bound': summary.loc['25%'] - 1.5 * (summary.loc['75%'] - summary.loc['25%']),
            'upper_bound': summary.loc['75%'] + 1.5 * (summary.loc['75%'] - summary.loc['25%'])
        })

    @st.cache_data                                                                                                          
    def std_approach(self, alpha: int = 3):
        """Applies a standard deviation method to define outlier thresholds."""
        df = st.session_state.data_standardized
        summary = df.describe()
        st.session_state.filter_ranges = pd.DataFrame({
            'feature': summary.columns,
            'lower_bound': summary.loc['mean'] - alpha * summary.loc['std'],
            'upper_bound': summary.loc['mean'] + alpha * summary.loc['std']
        })

    @st.cache_data
    def f1(self, z: pd.Series, beta_1: int, beta_2: int) -> pd.Series:
        """Detects outliers based on specified z-score thresholds."""
        anomaly_mask = (z <= -beta_1) | (z >= beta_2)
        return anomaly_mask

    @st.cache_data
    def f2(self, z: pd.Series, beta_1: int, beta_2: int, gamma: int = 3) -> pd.Series:
        """Combines z-score and isolation forest methods for outlier detection."""
        mask_f1 = self.f1(z, gamma * beta_1, gamma * beta_2)
        mask_iforest = self.isolation_forest(z)
        return mask_f1 & mask_iforest

    @st.cache_data
    def gamma_outlier(self, alphas: int = 6, alphak: int = 30):
        """Detects outliers using skewness and kurtosis thresholds with flexible gamma adjustments."""
        df = st.session_state.data_standardized
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
            filtered_data = data[mask]
            filter_ranges.append({
                'feature': column,
                'lower_bound': filtered_data.min(),
                'upper_bound': filtered_data.max()
            })
        st.session_state.filter_ranges = pd.DataFrame(filter_ranges)

    @st.cache_data
    def isolation_forest(self, z: pd.Series) -> pd.Series:
        """Detects outliers using the Isolation Forest algorithm."""
        iso_forest = IsolationForest(random_state=42)
        return pd.Series(iso_forest.fit_predict(z.values.reshape(-1, 1)), index=z.index) == 1
    

    @st.cache_data
    def boxplot(self, data, column, box_color='cyan', whisker_color='blue', median_color='orange', outlier_color='red', bound_color='purple'):
        
        np_array_data = np.array(data[column])  
        sorted_data = np.sort(np_array_data)

        min_val = np.min(sorted_data)
        q1 = np.percentile(sorted_data, 25)
        median = np.median(sorted_data)
        q3 = np.percentile(sorted_data, 75)
        max_val = np.max(sorted_data)

        df_ranges = st.session_state.filter_ranges_original
        row = df_ranges[df_ranges['feature'] == column]

        lower_bound = row['lower_bound'].values[0]
        upper_bound = row['upper_bound'].values[0]

        outliers = sorted_data[(sorted_data < lower_bound) | (sorted_data > upper_bound)]
        
        fig, ax = plt.subplots(figsize=(8, 4))  

        ax.add_patch(plt.Rectangle((q1, 0.2), q3 - q1, 0.6, color=box_color, edgecolor=whisker_color))

        ax.plot([median, median], [0.2, 0.8], color=median_color, linewidth=2, label='Median')
        ax.plot([min_val, q1], [0.5, 0.5], color=whisker_color, linewidth=2, label='Whiskers')  
        ax.plot([q3, max_val], [0.5, 0.5], color=whisker_color, linewidth=2)  
        ax.plot([min_val, min_val], [0.2, 0.8], color=whisker_color, linestyle='--', linewidth=1) 
        ax.plot([max_val, max_val], [0.2, 0.8], color=whisker_color, linestyle='--', linewidth=1) 
        ax.plot([lower_bound, lower_bound], [0.2, 0.8], color=bound_color, linewidth=2, linestyle='--', label='Lower Bound')
        ax.plot([upper_bound, upper_bound], [0.2, 0.8], color=bound_color, linewidth=2, linestyle='--', label='Upper Bound')

        if len(outliers) > 0:
            ax.scatter(outliers, [0.5] * len(outliers), color=outlier_color, label='Outliers')

        ax.set_yticks([]) 
        ax.set_xticks([lower_bound, median, upper_bound])
        ax.set_xticklabels([f'Lower: {lower_bound}', f'Median: {median}', f'Upper: {upper_bound}'])
        ax.set_title(f'Boxplot for {column}')
        ax.set_xlabel('Values')
        ax.legend()  

        st.pyplot(fig)
        plt.close(fig) 



