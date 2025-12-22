import pandas as pd
import logging
from abc import ABC, abstractmethod
logging.basicConfig(level= logging.INFO, format= '%(asctime)s - %(levelname)s - %(message)s')

class FeatureBinningStrategy(ABC):
    @abstractmethod
    def bin(self,df: pd.DataFrame,column_name:str)-> pd.DataFrame:
        pass

class CustomBinningStrategy(FeatureBinningStrategy):
    def __init__(self,bin_mapping):
        self.bin_mapping= bin_mapping
        logging.info(f" Custom binning initialized with mapping {bin_mapping}")

    def bin(self,df,column_name):
        def custom_binning(score):
            for bin_label,bin_range in self.bin_mapping.items():
                if  bin_range[0] <= score < bin_range[1]:
                    return bin_label
            raise ValueError(f"Score {score} is not in any bin range")
        try:
            df[f"{column_name}_binned"]= df[column_name].apply(custom_binning)
            del df[column_name]
            logging.info(f"Custom binning applied to {column_name} successfully ")
        except ValueError as e:
            logging.info(f"Could not bin column {column_name} with error {e}")
        return df
    

            

        

        