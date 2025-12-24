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
        df[scale_columns] = self.standard_scaler.fit_transform(df[scale_columns])
        self.fitted= True
        logging.info(f" Standard scaling fitted and transformed for columns: {scale_columns} successfully")
        return df
    
    def transform_features(self,df,scale_columns):
        df[scale_columns]= self.get_scaler().transform(df[scale_columns])
        logging.info(f"Transformed {scale_columns} successfully")
        return df
    
    def get_scaler(self):
        return self.standard_scaler
    
class MinMaxScaling(FeatureScalingStrategy):
    def __init__(self):
        self.min_max_scaler= MinMaxScaler().set_output(transform='pandas')
        self.fitted= False
        logging.info(f"MinMax scaling initialized")
    
    def scale_features(self, train_df, scale_columns):
        for column in scale_columns:
            train_df[column] =  self.min_max_scaler.fit_transform(train_df[[column]])
            self.fitted= True
            logging.info(f" MinMax scaling applied to {column} successfully")
        return train_df
    
    def transform_features(self,test_df,scale_columns):
        for column in scale_columns:
            test_df[column]= self.get_scaler().transform(test_df[[column]])
            logging.info(f"Transformed {column} successfully")
        return test_df
    
    def get_scaler(self):
        return self.min_max_scaler