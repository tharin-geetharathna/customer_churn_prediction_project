import pandas as pd
import logging
from abc import ABC, abstractmethod
logging.basicConfig(level= logging.INFO, format= '%(asctime)s - %(levelname)s - %(message)s')
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

class FeatureEncodingStrategy(ABC):
    @abstractmethod
    def encode(self,df: pd.DataFrame)-> pd.DataFrame:
        pass

class Ordinalencoding(FeatureEncodingStrategy):
    def __init__(self,ordinal_mapping):
        self.ordinal_mapping= ordinal_mapping
        logging.info(f" Ordinal encoding initalized with mapping. ")

    def encode(self, df):
        for col,mapping in self.ordinal_mapping.items():
            df[col]= df[col].map(mapping)
            logging.info(f" Ordinal encoding applied to {col} successfully")
            encoded_values= df[col].value_counts().to_dict()
            logging.info(f" Values after encoding {encoded_values}")
        return df
    
class NominalEncoding(FeatureEncodingStrategy):
    def __init__(self,nominal_columns):
        self.nominal_columns = nominal_columns
        self.ohe= OneHotEncoder(sparse_output= False,handle_unknown='ignore',drop='first').set_output(transform='pandas')
        logging.info(f" One hot encoding initialized")

    def encode(self,df):
        for column in self.nominal_columns:
            ohe_transformed_df= self.ohe.fit_transform(df[[column]])
            del df[column]
            df= pd.concat([df,ohe_transformed_df],axis=1)
            logging.info(f" One hot encoding applied to {column} successfully")
        logging.info(f"One hot encoding successfully applied to nominal columns")
        return df

    


        

