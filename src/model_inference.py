import logging,os,sys
import pandas as pd
from pathlib import Path
import joblib
import json
logging.basicConfig(level=logging.INFO, format= '%(asctime)s - %(levelname)s - %(message)s')

sys.path.append(os.path.join(os.path.dirname(__file__),'..','utils'))
sys.path.append(os.path.join(os.path.dirname(__file__),'..','src'))
from handle_missing_values import DropFeatures
from config import get_data_paths,get_columns,get_scaling_config

class ModelInference:
    def __init__(self,model_path):
        self.model_path= model_path
        self.data_paths = get_data_paths()
        self.scaling_config= get_scaling_config()
        self.columns = get_columns()
        self.model= None
        self.binning_handler = None
        self.nominal_encoder= None
        self.ordinal_encoder= None
        self.scaler= None
        self.feature_order =None


    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model is not found at the location {self.model_path}")
        self.model = joblib.load(self.model_path)
        logging.info(f"Model succesfully loaded")
    
    def load_preprocessors(self):
        if not os.path.exists(self.data_paths['binning_handler'])\
            or not os.path.exists(self.data_paths['nominal_encoder_handler'])\
            or not os.path.exists(self.data_paths['ordinal_encoder_handler'])\
            or not os.path.exists(self.data_paths['scaling_handler'])\
            or not os.path.exists(self.data_paths['feature_order']):
            raise FileNotFoundError(f"Preprocessors are not found at the location {self.data_paths}")
        
        self.binning_handler =joblib.load(self.data_paths['binning_handler'])
        self.nominal_encoder_handler =joblib.load(self.data_paths['nominal_encoder_handler'])
        self.ordinal_encoder_handler =joblib.load(self.data_paths['ordinal_encoder_handler'])
        self.scaling_handler =joblib.load(self.data_paths['scaling_handler'])
        with open(self.data_paths['feature_order']) as file:
            self.feature_order = json.load(file)

        logging.info("Loaded preprocessors successfully")

        
    def preprocess_input(self,data:dict) -> pd.DataFrame:
        df= pd.DataFrame([data])
        df = self.binning_handler.bin(df,column_name ='CreditScore')
        df= self.nominal_encoder_handler.encode(df)
        df= self.ordinal_encoder_handler.encode(df)
        df= DropFeatures(drop_columns= self.columns['drop_columns']).drop_features(df)
        for col in self.feature_order:
            if col not in df.columns:
                df[col] = 0
        df= df[self.feature_order]
        df= self.scaling_handler.transform_features(df,scale_columns= self.scaling_config['columns_to_scale'])
        logging.info("Preprocessed the input successfully")
        return df
    
    def predict(self,data:dict):
        X= self.preprocess_input(data)
        prediction = self.model.predict(X)
        probability= self.model.predict_proba(X)[0,1]
        return {
            "prediction": int(prediction[0]),
            "probability": float (probability)
        }



        