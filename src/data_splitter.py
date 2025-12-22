import pandas as pd
import logging 
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Tuple
logging.basicConfig(level= logging.INFO, format= '%(asctime)s - %(levelname)s - %(message)s')

class DataSplitter(ABC):
    @abstractmethod
    def split_data(self,df: pd.DataFrame,target_column:str)-> Tuple[pd.DataFrame,pd.DataFrame,pd.Series,pd.Series]:
        pass

class TrainTestSplit(DataSplitter):
    def __init__(self,test_size= 0.2,random_state=42):
        self.test_size= test_size
        self.random_state= random_state
        logging.info(f" Traint test split initialized with test size {test_size} and random state {random_state}")
    
    def split_data(self, df, target_column):
        Y= df[target_column]
        X= df.drop(columns=[target_column])
        X_train,X_test,y_train,y_test= train_test_split(X,Y,test_size= self.test_size,random_state= self.random_state)
        logging.info(f" Train test split applied successfully")

        X_train.to_csv("artifacts/Splits/X_train.csv", index=False)
        X_test.to_csv("artifacts/Splits/X_test.csv", index=False)
        y_train.to_csv("artifacts/Splits/y_train.csv", index=False)
        y_test.to_csv("artifacts/Splits/y_test.csv", index=False)
        logging.info("Data splits saved to artifacts/Splits as CSV")
        return X_train,X_test,y_train,y_test
        
