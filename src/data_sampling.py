import pandas as pd
import logging
from imblearn.over_sampling import SMOTE
from typing import Tuple
from abc import ABC, abstractmethod
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataSamplingStrategy(ABC):
    @abstractmethod
    def sample_data(self,X_train: pd.DataFrame,y_train: pd.Series)-> Tuple[pd.DataFrame,pd.Series]:
        pass
    
class SMOTESampling(DataSamplingStrategy):
    def __init__(self,random_state= 42):
        self.random_state= random_state
        self.smote= SMOTE(random_state= self.random_state)
        logging.info(f"SMOTE sampling initialized with random state {self.random_state} and imported training and testing data for over sampling")

    def sample_data(self,X_train,y_train):
        X_train_resampled,y_train_resampled= self.smote.fit_resample(self.X_train,self.y_train)
        X_train_resampled.to_csv("artifacts/Splits/X_train_resampled.csv", index=False)
        y_train_resampled.to_csv("artifacts/Splits/y_train_resampled.csv",index= False)
        logging.info(f"SMOTE sampling applied successfully")
        return X_train_resampled,y_train_resampled

