import os 
from abc import ABC, abstractmethod
import joblib
from datetime import datetime
import logging
logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(levelname)s - %(message)s")
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

class BaseModelBuilder(ABC):
    def __init__(
                self,
                model_name= None,
                **kwargs
                ):
        self.model_name= model_name
        self.model= None
        self.model_params = kwargs

    @abstractmethod
    def build_model(self):
        pass

    def save_model(self,file_path):
        if self.model is None:
            raise ValueError("Model is not built yet")
        
        joblib.dump(self.model,file_path)
        logging.info(f"Model saved to {file_path}")

    def load_model(self,file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model is not found at {file_path}")
        self.model= joblib.load(file_path)
        logging.info(f"Model loaded successfully from {file_path}")
        return self.model

class RandomforestModelBuilder(BaseModelBuilder):
    def __init__(self,**kwargs):
        default_params={
                        'n_estimators': 200,
                        'max_depth': 10,
                        'min_samples_split': 2,
                        'min_samples_leaf':1,
                        'random_state': 42
                        
                        }
        default_params.update(kwargs)
        super().__init__(model_name='RandomForest',**default_params)
    
    def build_model(self):
        self.model = RandomForestClassifier(**self.model_params)
        logging.info("RandomForest model built successfully")
        return self.model
    
class XGBoostModelBuilder(BaseModelBuilder):
    def __init__(self,**kwargs):
         default_params={
                        'n_estimators': 100,
                        'max_depth': 10,
                        'min_samples_split':2,
                        'min_samples_leaf':1,
                        'random_state': 42
                        }
         default_params.update(kwargs)
         super().__init__(model_name='XGBoost',**default_params)
    
    def build_model(self):
        self.model= XGBClassifier(**self.model_params)
        logging.info("XGBoost model built successfully")
        return self.model
