import pandas as pd
import streamlit as st
from dataclasses import dataclass

@dataclass
class Utilitis:
    
    def non_numerical_data_cleaner(self, df:pd.DataFrame)->pd.DataFrame:
        return df.select_dtypes(include=['number'])

