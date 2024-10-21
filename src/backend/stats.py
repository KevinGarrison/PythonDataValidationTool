import pandas as pd
import streamlit as st
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

@dataclass
class Statistics:
    
    @st.cache_data
    def feature_stats_summarize(self):
        df = st.session_state.data
        st.session_state.stats_summarize = df.describe()
        
    # Interquartile Range approach for normally distributed data
    @st.cache_data
    def iqr_approach(self):
        df = st.session_state.data
        summary = df.describe()

        st.session_state.filter_ranges = pd.DataFrame({
            'feature': summary.columns,
            'lower_bound': summary.loc['25%'] - 1.5 * (summary.loc['75%'] - summary.loc['25%']),
            'upper_bound': summary.loc['75%'] + 1.5 * (summary.loc['75%'] - summary.loc['25%']) 
        })

    @st.cache_data
    def std_approach(self, alpha: int = 3):
        df = st.session_state.data
        summary = df.describe()

        st.session_state.filter_ranges = pd.DataFrame({
            'feature': summary.columns,
            'lower_bound': summary.loc['mean'] - alpha * summary.loc['std'],
            'upper_bound': summary.loc['mean'] + alpha * summary.loc['std']
        })

    # Clustering approach for unknown distribution datasets
    @st.cache_data
    def clustering_approach(self, threshold: float = 0.1, n_clusters: int = 2, n_neighbors: int = 20, approach: int = 0): # TODO Research on treshhold determination
        df = st.session_state.data
        if approach == 0:
            # Using K-Means
            kmeans = KMeans(n_clusters=n_clusters)
            df['cluster'] = kmeans.fit_predict(df)
            
            df['distance_to_centroid'] = kmeans.transform(df).min(axis=1)
            normal_data_kmeans = df[df['distance_to_centroid'] <= threshold]
            
            st.session_state.filter_ranges = pd.DataFrame({
                'feature': df.columns[:-2],  # Exclude 'distance_to_centroid' and 'cluster'
                'lower_bound': normal_data_kmeans.min(),
                'upper_bound': normal_data_kmeans.max()
            })
        elif approach == 1:
            # Using Local Outlier Factor (LOF)
            lof = LocalOutlierFactor(n_neighbors=n_neighbors) # TODO Research on neighbors determination
            df['is_anomaly_lof'] = lof.fit_predict(df)
            
            normal_data_lof = df[df['is_anomaly_lof'] != -1]
            
            st.session_state.filter_ranges = pd.DataFrame({
                'feature': df.columns[:-3],  # Exclude 'distance_to_centroid', 'cluster', 'is_anomaly_lof'
                'lower_bound': normal_data_lof.min(),
                'upper_bound': normal_data_lof.max()
            })

    # Isolation Forest approach
    @st.cache_data      
    def isolation_forest_approach(self, threshold: float = 0.1): # TODO Research on treshhold determination
        df = st.session_state.data
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        iso_forest.fit(df)

        anomaly_scores = iso_forest.decision_function(df)
        df['anomaly_score'] = anomaly_scores
        
        normal_data = df[anomaly_scores > threshold]

        st.session_state.filter_ranges = pd.DataFrame({
            'feature': df.columns[:-1],  # Exclude the anomaly_score column
            'lower_bound': normal_data.min(),
            'upper_bound': normal_data.max()
        })

