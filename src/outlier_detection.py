import logging
from abc import ABC, abstractmethod
import pandas as pd
from typing import List
logging.basicConfig(level= logging.INFO, format= '%(asctime)s - %(levelname)s - %(message)s')


class OutlierDetectionStrategy(ABC):
    @abstractmethod
    def detect_outliers(self,df: pd.DataFrame, columns : List[str])-> pd.DataFrame:
        pass

class IQROutlierDetection(OutlierDetectionStrategy):
    def detect_outliers(self,df,columns):
        outliers_df= pd.DataFrame(False, index= df.index, columns = columns)
        for col in columns:
            Q1= df[col].quantile(0.25)
            Q3= df[col].quantile(0.75)
            IQR= Q3-Q1
            upper_bound = Q3 + 1.5*IQR
            lower_bound = Q1 -1.5*IQR
            outliers_df[col]= (df[col]<lower_bound)| (df[col]> upper_bound)
            outlier_count = outliers_df[col].sum()
            logging.info(f"{outlier_count} outliers detected in column {col}")
        total_outliers = outliers_df.values.sum()
        logging.info(f"Total {total_outliers} outliers detected")
        return outliers_df

class OutlierDetector():
    def __init__(self,strategy):
        self.strategy= strategy

    def detect_outliers(self,df,outlier_columns):
        return self.strategy.detect_outliers(df,outlier_columns)
    
    def handle_outliers(self, df, outlier_columns,method='remove'):
        outliers= self.detect_outliers(df,outlier_columns)
        outlier_count = outliers.sum(axis=1)
        rows_to_drop = outlier_count>=2
        indexes_with_2plus_outliers = outliers.index[rows_to_drop].tolist()
        logging.info(f"Indexes with >=2 outliers: {indexes_with_2plus_outliers}")
        logging.info(f"{rows_to_drop.sum()} outliers have been removed successfuly")
        return df[~rows_to_drop]




    