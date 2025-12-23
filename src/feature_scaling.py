import pandas as pd
import logging
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from typing import List
logging.basicConfig(level= logging.INFO, format= '%(asctime)s - %(levelname)s - %(message)s')

class FeatureScalingStrategy(ABC):
    @abstractmethod
    def scale_features(self,df: pd.DataFrame,scale_columns:List[str])-> pd.DataFrame:
        pass


class StandardScaling(FeatureScalingStrategy):
    def __init__(self):
        self.standard_scaler= StandardScaler().set_output(transform='pandas')
        self.fitted= False
        logging.info(f" Standard scaling initialized")

    def scale_features(self, df, scale_columns):
        for column in scale_columns:
            df[column] = self.standard_scaler.fit_transform(df[[column]])
            self.fitted= True
            logging.info(f" Standard scaling applied to {column} successfully")
        return df
    
    def get_scaler(self):
        return self.standard_scaler
    
class MinMaxScaling(FeatureScalingStrategy):
    def __init__(self):
        self.min_max_scaler= MinMaxScaler().set_output(transform='pandas')
        self.fitted= False
        logging.info(f"MinMax scaling initialized")
    
    def scale_features(self, df, scale_columns):
        for column in scale_columns:
            df[column] =  self.min_max_scaler.fit_transform(df[[column]])
            self.fitted= True
            logging.info(f" MinMax scaling applied to {column} successfully")
        return df
    
    def get_scaler(self):
        return self.min_max_scaler