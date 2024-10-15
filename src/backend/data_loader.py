import pandas as pd
import streamlit as st
from dataclasses import dataclass 

@dataclass
class DataLoader: # TODO: Modify the class for loading various formats 
    df: pd.DataFrame = None
    def load_data(self, uploaded_file) -> pd.DataFrame:
        if uploaded_file is not None:
            file_path = str(uploaded_file.name)
            if file_path[-4:] == ".csv":
                self.df = pd.read_csv(uploaded_file)
            elif (file_path[-4:] == ".xls") | (file_path[-4:] == ".xlsx"):
                self.df = pd.read_excel(uploaded_file)
            else:
                raise TypeError(f"The file has a wrong type {file_path.name}")
        else:
            raise ValueError("No file uploaded") 
    
    def data_clearer(self):
        self.df = None
