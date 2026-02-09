import os 
import sys
import logging
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict
import matplotlib.pyplot as plt
import joblib
import json
import warnings
import mlflow
warnings.filterwarnings('ignore')

logging.basicConfig(level= logging.INFO, format= '%(asctime)s - %(levelname)s - %(message)s')

sys.path.append(os.path.join(os.path.dirname(__file__),'..','src'))
from data_ingestion import DataIngestorCSV
from handle_missing_values import DropMissingValues,FillMissingValueStrategy,GenderImputer,DropFeatures
from outlier_detection import IQROutlierDetection,OutlierDetector
from feature_binning import CustomBinningStrategy
from feature_encoding import NominalEncoding,OrdinalEncoding
from feature_scaling import MinMaxScaling,StandardScaling
from data_sampling import SMOTESampling
from data_splitter import TrainTestSplit
sys.path.append(os.path.join(os.path.dirname(__file__),'..','utils'))
from config import get_data_paths,get_columns,get_missing_values_config,get_outlier_config,get_binning_config,get_encoding_config,get_scaling_config,get_splitting_config,get_training_config
from mlflow_utils import MLflowTracker, setup_autolog, create_mlflow_run_tags

def datapipeline(
        target_column: str = 'Exited',
        test_size: float= 0.2,
        force_rebuild: bool = False
        ) -> Dict[str, np.ndarray]:
     
    try:   
        # Initalize MLflow tracker 
        mlflow_tracker = MLflowTracker()
        run_tags = create_mlflow_run_tags(pipeline_name= 'data_pipeline',additional_tags={'target_column': target_column})
        run = mlflow_tracker.start_run(run_name= 'data_pipeline',tags= run_tags)

        data_paths= get_data_paths()
        columns = get_columns()
        missing_values_config= get_missing_values_config()
        outlier_config =  get_outlier_config()
        binning_config= get_binning_config()
        encoding_config =  get_encoding_config()
        scaling_config= get_scaling_config()
        splitting_config= get_splitting_config()
        training_config= get_training_config()

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
        df= ingestor.ingest(data_paths['raw_data'])
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

        print("4. Feature binning")
        binning_handler= CustomBinningStrategy(bin_mapping= binning_config['credit_score_binning'])
        df = binning_handler.bin(df,column_name='CreditScore')
        print(f"Shape of the data after feature_binning {df.shape}")
        print(f"Values in the column credit score binned : {df['CreditScore_binned'].value_counts()}")

        print("5. Feature encoding")
        nominal_encoding= NominalEncoding(nominal_columns= encoding_config['nominal_columns'])
        ordinal_encoding= OrdinalEncoding (ordinal_mapping= encoding_config['ordinal_mapping'])
        df= nominal_encoding.encode(df)
        df= ordinal_encoding.encode(df)
        print(f"Shape of the data after feature encoding {df.shape}")

        print("6. Feature dropping")
        feature_drop_handler= DropFeatures(drop_columns=columns['drop_columns'])
        df= feature_drop_handler.drop_features(df)

        print("7. Data Splitting")
        train_test_split= TrainTestSplit(test_size=training_config['test_size'],random_state=training_config['random_state'])
        X_train,X_test,y_train,y_test = train_test_split.split_data(df,target_column=columns['target_columns'])


        print("8. Feature scaling")
        standard_scaler= StandardScaling()
        X_train= standard_scaler.scale_features(X_train,scale_columns= scaling_config['columns_to_scale'] )
        print("Training data after scaled\n",X_train.head())
        X_test= standard_scaler.transform_features(X_test,scale_columns= scaling_config['columns_to_scale'] )
        print("Testing data after scaled\n",X_test.head())

        print("9. Sampling")
        smote_sampling = SMOTESampling(random_state= training_config['random_state'])
        X_train,y_train= smote_sampling.sample_data(X_train,y_train)
        print(f"Shape of the train data after sampling {X_train.shape}")
        print(f"Shape of the test data after sampling {y_train.shape}")

        dataset_info ={
            'dataset_rows': len(df),
            'training_rows': len(X_train),
            'test_rows': len(X_test),
            'number_of_features': df.shape[1],
            'feature_names' : df.columns.tolist(),
            'dropped_features_count': len(columns['drop_columns']),
            'test_size' : training_config['test_size'],
            'random_state': training_config['random_state'],
            'missing_values_handled': True,
            'outliers_handled': True,
            'feature_binning_handled': True,
            'feature_encoding_handled': True,
            'feature_scaling_handled': True
        }

        mlflow_tracker.log_data_pipeline_metrics(dataset_info= dataset_info)


        # Saving the data splits
        os.makedirs('artifacts/data_splits', exist_ok=True)
        X_train.to_csv("artifacts/data_splits/X_train.csv", index=False)
        X_test.to_csv("artifacts/data_splits/X_test.csv", index=False)
        y_train.to_csv("artifacts/data_splits/y_train.csv", index=False)
        y_test.to_csv("artifacts/data_splits/y_test.csv", index=False)
        logging.info("Data splits saved after preprocessing to artifacts/data_splits as CSV")

        ### Dumping the preprocessors
        os.makedirs(data_paths['preprocessors_dir'],exist_ok=True)
        joblib.dump(binning_handler,data_paths['binning_handler'])
        joblib.dump(nominal_encoding,data_paths['nominal_encoder_handler'])
        joblib.dump(ordinal_encoding,data_paths['ordinal_encoder_handler'])
        joblib.dump(standard_scaler,data_paths['scaling_handler'])
        logging.info("Preprocessors saved successfully")

        #Log preprocessors to MLflow
        mlflow.log_artifact(local_path= data_paths['binning_handler'],artifact_path= 'preprocessors')
        mlflow.log_artifact(local_path= data_paths['nominal_encoder_handler'],artifact_path= 'preprocessors')
        mlflow.log_artifact(local_path= data_paths['ordinal_encoder_handler'],artifact_path= 'preprocessors')
        mlflow.log_artifact(local_path= data_paths['scaling_handler'],artifact_path= 'preprocessors')
        mlflow.log_artifact(local_path= data_paths['feature_order'],artifact_path= 'preprocessors')

        # Log splits to MLflow
        mlflow.log_artifact(local_path = data_paths['X_train'],artifact_path = 'splits')
        mlflow.log_artifact(local_path = data_paths['X_test'],artifact_path = 'splits')
        mlflow.log_artifact(local_path = data_paths['y_train'],artifact_path = 'splits')
        mlflow.log_artifact(local_path = data_paths['y_test'],artifact_path = 'splits')

        feature_order = X_train.columns.tolist()
        with open(data_paths['feature_order'],'w') as file:
            json.dump(feature_order,file)
            logging.info(f"Feature order saved to {data_paths['feature_order']}")
    except Exception as e:
        logging.error(f"Error logging splits to MLflow. {e}")
        if 'mlflow_tracker' in locals():
            mlflow_tracker.end_run()
        raise
        
datapipeline()