import logging
from abc import ABC, abstractmethod
import pandas as pd
logging.basicConfig(level= logging.INFO, format= '%(asctime)s - %(levelname)s - %(message)s')


class OutlierDetectionStrategy(ABC):
    @abstractmethod
    def detect_outliers(self,df: pd.DataFrame, columns : List)-> pd.DataFrame:
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
        total_outliers = outliers_df.sum()
        logging.info(f"Total {total_outliers} outliers detected")
        return outliers_df

class OutlierDetector():
    def __init(self,strategy):
        self.strategy= strategy

    def detect_outliers(self,df,numerical_columns):
        return self.strategy.detect_outliers(df,numerical_columns)
    
    def handle_outliers(self, df, numerical_columns,method='remove'): 
        outliers= self.detect_outliers(df,numerical_columns)
        outlier_count = outliers.any(axis=1).sum()
        rows_to_drop = outlier_count>=2
        return df[~rows_to_drop]


    