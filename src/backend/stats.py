import pandas as pd
import streamlit as st
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
        

@dataclass
class Statistics:
    
    @st.cache_data
    def feature_stats_summarize():
        df = st.session_state.data
        st.session_state.stats_summarize =  df.describe()
        
    # Inter Quartil Range approach for normal distributed data for larger data sets
    @st.cache_data
    def iqr_approach():
        df = st.session_state.data
        # Calculate the statistical summary for each feature
        summary = df.describe()

        st.session_state.filter_ranges = pd.DataFrame({
            'feature': summary.columns,
            'lower_bound': summary.loc['25%'] - 1.5 * (summary.loc['75%'] - summary.loc['25%']),
            'upper_bound': summary.loc['75%'] + 1.5 * (summary.loc['75%'] - summary.loc['25%']) 
        })

    @st.cache_data
    def std_approach(alpha:int=3):
        df = st.session_state.data
        alpha = alpha 
        # Calculate the statistical summary for each feature        
        summary = df.describe()

        st.session_state.filter_ranges = pd.DataFrame({
            'feature': summary.columns,
            'lower_bound': summary.loc['mean'] - alpha * summary.loc['std'],
            'upper_bound': summary.loc['mean'] + alpha * summary.loc['std']
        })

    # Clustering approach for unkwown distribution datasets > 1000 observations # TODO research information about CLustering approach with numerical data       
    @st.cache_data
    def clustering_approach(threshold: float=0.1, n_clusters: int=2, n_neighbors: int=20, approach:list=[0,1]):
        df = st.session_state.data
        if approach == 0:
            # Using K-Means
            kmeans = KMeans(n_clusters=n_clusters)
            df['cluster'] = kmeans.fit_predict(df)
            
            # Calculate distance of each point to the closest centroid
            df['distance_to_centroid'] = kmeans.transform(df).min(axis=1)
            
            # Filter out normal points (points not considered anomalies by K-Means)
            normal_data_kmeans = df[df['distance_to_centroid'] <= threshold]
            
            # Calculate recommended filter ranges based on the filtered normal data
            st.session_state.filter_ranges = pd.DataFrame({
                'feature': df.columns[:-2],  # Ignore 'distance_to_centroid' and 'cluster' columns
                'lower_bound': normal_data_kmeans.min(),
                'upper_bound': normal_data_kmeans.max()
            })
        elif approach == 1:
            # Using Local Outlier Factor (LOF)
            lof = LocalOutlierFactor(n_neighbors=n_neighbors)
            df['is_anomaly_lof'] = lof.fit_predict(df)
                        
            # Filter out normal points (points not considered anomalies by LOF)
            normal_data_lof = df[df['is_anomaly_lof'] != -1]
            
            # Calculate recommended filter ranges based on the filtered normal data
            st.session_state.filter_ranges = pd.DataFrame({
                'feature': df.columns[:-3],  # Ignore 'distance_to_centroid', 'cluster', and 'is_anomaly_lof'
                'lower_bound': normal_data_lof.min(),
                'upper_bound': normal_data_lof.max()
            })

    # Isolation Forest approach for unkwown distribution datasets > 1000 observations
    @st.cache_data           
    def if_approach(treshold:float=0.1):
        df = st.session_state.data
        # Fit the Isolation Forest model
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        iso_forest.fit(df)

        # Get the anomaly scores for each data point
        anomaly_scores = iso_forest.decision_function(df)

        # Add the scores as a column in the dataset
        df['anomaly_score'] = anomaly_scores

        # Determine normal range recommendations based on score thresholds
        threshold = treshold 

        # Filter out points that have anomaly scores below the threshold
        normal_data = df[anomaly_scores > threshold]

        # Calculate recommended filter ranges based on this filtered "normal" data
        st.session_state.filter_ranges = pd.DataFrame({
            'feature': df.columns[:-1],  # Ignore the anomaly_score column
            'lower_bound': normal_data.min(),
            'upper_bound': normal_data.max()
        })
    
    @st.cache_data
    def run_approach(approach:str='IQR'):
        if approach == 'IQR':
            iqr_approach()
    

        



