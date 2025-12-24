import os 
import sys
import logging
from typing import Dict
import pandas as pd
import json

logging.basicConfig(level= logging.INFO, format= '%(asctime)s - %(levelname)s - %(message)s')

sys.path.append(os.path.join(os.path.dirname(__file__),'..','src'))
sys.path.append(os.path.join(os.path.dirname(__file__),'..','utils'))
sys.path.append(os.path.join(os.path.dirname(__file__),'..','pipelines'))
from data_pipeline import datapipeline
from model_building import RandomforestModelBuilder,XGBoostModelBuilder
from model_training import ModelTraining
from config import get_training_config,get_data_paths,get_model_config,get_columns

def training_pipeline(
                    data_pathsdata_path:str = 'data/raw/ChurnModelling.csv',
                    random_state: int =42,
                    rf_file_path : str ='artifacts/models/random_forest.joblib',
                    xgb_file_path : str ='artifacts/models/xgboost.joblib'
                    ):
    data_paths= get_data_paths()
    columns = get_columns()
    training_config= get_training_config()
    model_config = get_model_config()
    os.makedirs(data_paths['model_artifacts_dir'],exist_ok=True)

    if not os.path.exists(data_paths['X_train']) or \
            not os.path.exists(data_paths['X_test']) or \
            not os.path.exists(data_paths['y_train']) or \
            not os.path.exists(data_paths['y_test']):
        datapipeline()
    else:
        print("Loading data artifacts from data pipeline")
    X_train = pd.read_csv(data_paths['X_train'])
    X_test = pd.read_csv(data_paths['X_test'])
    y_train = pd.read_csv(data_paths['y_train'])
    y_test = pd.read_csv(data_paths['y_test'])

    model_builder_rf= RandomforestModelBuilder(**model_config['model_types']['random_forest'])
    model_builder_xgb= XGBoostModelBuilder(**model_config['model_types']['gradient_boosting'])
    if not os.path.exists(rf_file_path):
        rf_model = model_builder_rf.build_model()
        rf_model,train_score = ModelTraining.train_model(
            model= rf_model,
            X_train = X_train,
            y_train=y_train
        )
        logging.info(f"RF Model trained with a training score of {train_score}")
        model_builder_rf.save_model(rf_file_path)
    

    rf_model = model_builder_rf.load_model(rf_file_path)
    print(model_builder_rf.model_params)

    if not os.path.exists(xgb_file_path):
        xgb_model = model_builder_xgb.build_model()
        xgb_model,train_score= ModelTraining.train_model(
            model= xgb_model,
            X_train = X_train,
            y_train=y_train
        )
        logging.info(f"XGBoost Model trained with a training score of {train_score}")
        model_builder_xgb.save_model(xgb_file_path)
    
    xgb_model= model_builder_xgb.load_model(xgb_file_path)
    print(model_builder_xgb.model_params)

training_pipeline()

    
