import pandas as pd, numpy as np
import streamlit as st
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from scipy.stats import skew, kurtosis

@dataclass
class Statistics:
    """A data analysis class that provides methods for summarizing data, detecting outliers, 
       and normalizing features using various statistical and machine learning approaches."""

    @st.cache_data
    def feature_stats_summarize(self) -> pd.DataFrame:
        """Summarizes statistical features of the dataset."""
        try:
            df = st.session_state.get('data')
            if df is None or df.empty:
                st.warning("The data is empty. Please load data first.")
                return pd.DataFrame()
            return df.describe()
        except Exception as e:
            st.error(f"An error occurred while summarizing statistics: {e}")
            return pd.DataFrame()

    @st.cache_data
    def iqr_approach(self) -> pd.DataFrame:
        """Applies the IQR method to calculate feature bounds for outlier detection."""
        df = st.session_state.get('data')
        summary = df.describe()
        return pd.DataFrame({
            'feature': summary.columns,
            'lower_bound': summary.loc['25%'] - 1.5 * (summary.loc['75%'] - summary.loc['25%']),
            'upper_bound': summary.loc['75%'] + 1.5 * (summary.loc['75%'] - summary.loc['25%'])
        })

    @st.cache_data
    def std_approach(self, alpha: int = 3) -> pd.DataFrame:
        """Applies a standard deviation method to define outlier thresholds."""
        df = st.session_state.get('data')
        summary = df.describe()
        return pd.DataFrame({
            'feature': summary.columns,
            'lower_bound': summary.loc['mean'] - alpha * summary.loc['std'],
            'upper_bound': summary.loc['mean'] + alpha * summary.loc['std']
        })

    @st.cache_data
    def normalize_numerical_data(self) -> pd.DataFrame:
        """Standardizes numerical data using Z-score normalization."""
        try:
            df = st.session_state.get('data')
            scaler = StandardScaler()
            return scaler.fit_transform(df)
        except Exception as e:
            st.error(f"An error occurred during normalization: {e}")
            return pd.DataFrame()

    @st.cache_data
    def f1(self, z: pd.Series, beta_1: int, beta_2: int) -> pd.Series:
        """Detects outliers based on specified z-score thresholds."""
        anomaly_mask = (z <= -beta_1) | (z >= beta_2)
        return anomaly_mask.astype(int)

    @st.cache_data
    def f2(self, z: pd.Series, beta_1: int, beta_2: int, gamma: int = 3) -> pd.Series:
        """Combines z-score and isolation forest methods for outlier detection."""
        mask_f1 = self.f1(z, gamma * beta_1, gamma * beta_2)
        mask_iforest = self.isolation_forest(z)
        return mask_f1 & mask_iforest

    @st.cache_data
    def gamma_outlier(self, alphas: int = 6, alphak: int = 30) -> pd.DataFrame:
        """Detects outliers using skewness and kurtosis thresholds with flexible gamma adjustments."""
        df = st.session_state.get('data_standardized')
        if df is None:
            st.warning("Standardized data not found. Please normalize data first.")
            return pd.DataFrame()

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
        return pd.DataFrame(filter_ranges)

    @st.cache_data
    def isolation_forest(self, z: pd.Series) -> pd.Series:
        """Detects outliers using the Isolation Forest algorithm."""
        iso_forest = IsolationForest(random_state=42)
        return pd.Series(iso_forest.fit_predict(z.values.reshape(-1, 1)), index=z.index) == 1
