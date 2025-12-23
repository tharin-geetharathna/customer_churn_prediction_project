import os 
import sys
import logging
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict
import matplotlib.pyplot as plt
import json

logging.basicConfig(level= logging.INFO, format= '%(asctime)s - %(levelname)s - %(message)s')

sys.path.append(os.path.join(os.path.dirname(__file__),'..','src'))
from data_ingestion import DataIngestorCSV
from handle_missing_values import DropMissingValues,FillMissingValueStrategy,GenderImputer
from outlier_detection import IQROutlierDetection,OutlierDetector
from feature_binning import CustomBinningStrategy
from feature_encoding import NominalEncoding,Ordinalencoding
from feature_scaling import MinMaxScaler
from data_sampling import SMOTESampling
from data_splitter import TrainTestSplit
sys.path.append(os.path.join(os.path.dirname(__file__),'..','utils'))
from config import get_data_paths,get_columns,get_missing_values_config,get_outlier_config,get_binning_config,get_encoding_config,get_scaling_config,get_splitting_config

def datapipeline(
        data_path: str= 'data/raw/ChurnModelling.csv',
        target_column: str = 'Exited',
        test_size: float= 0.2,
        force_rebuild: bool = False
        ) -> Dict[str, np.ndarray]:
    
    data_paths= get_data_paths()
    columns = get_columns()
    missing_values_config= get_missing_values_config()
    outlier_config =  get_outlier_config()
    binning_config= get_binning_config()
    encoding_config =  get_encoding_config()
    scaling_config= get_scaling_config()
    splitting_config= get_splitting_config()

    print("1. Data ingestion")
    artificats_dir = os.path.join(os.path.dirname(__file__),'..',data_paths['data_artifacts_dir'])
    X_train_path = os.path.join('data_artifacts_dir','X_train.csv')
    X_test_path = os.path.join ('data_artifacts_dir','X_test.csv')
    y_train_path = os.path.join('data_artifacts_dir','y_train.csv')
    y_test_path = os.path.join('data_artifacts_dir','y_test.csv')

    if os.path.exists(X_train_path) and \
        os.path.exists(X_test_path) and \
        os.path.exists(y_train_path) and \
        os.path.exists(y_test_path):
        X_train=pd.read_csv(X_train_path)
        X_test= pd.read_csv(X_test_path)
        y_train= pd.read_csv(y_train_path)
        y_test= pd.read_csv(y_test_path)

    ingestor=  DataIngestorCSV()
    df= ingestor.ingest(data_path)
    print(f"Shape of the data after ingestion { df.shape}")

    print("2. Handling missing values")
    if not os.path.exists(os.path.join(os.path.dirname(__file__),'..','Data','Processed','temporary_imputed.csv')):
        critical_handler = DropMissingValues(critical_columns=columns['critical_columns'])
        age_handler= FillMissingValueStrategy(
                    imputing_column= 'Age',
                    method='mean'
                    )
        gender_handler= FillMissingValueStrategy(
                    imputing_column= 'Gender',
                    is_customer_imputer=True,
                    custom_imputer=GenderImputer()
                    )
        df= critical_handler.handle_missing_values(df)
        df= age_handler.handle_missing_values(df)
        df= gender_handler.handle_missing_values(df)
        df.to_csv('data/processed/temporary_imputed.csv',index=False)
        
    df= pd.read_csv("data/processed/temporary_imputed.csv")
    print(f"Shape of the data after handling missing values {df.shape}")
    print("3. Handling outliers")
    outlier_detector = OutlierDetector(IQROutlierDetection())
    df= outlier_detector.handle_outliers(df,outlier_columns= columns['outlier_columns'])
    print(f"Shape of the data after handling outliers {df.shape}")




    
datapipeline()