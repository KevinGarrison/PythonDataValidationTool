import pandas as pd
import streamlit as st
from dataclasses import dataclass 

@dataclass
class DataLoader():

    @st.cache_data
    def load_data(df:pd.DataFrame) -> pd.DataFrame:
        if df is not None and not df.empty:
            return df
        else:
            raise ValueError("The DataFrame is either None or empty.") 
