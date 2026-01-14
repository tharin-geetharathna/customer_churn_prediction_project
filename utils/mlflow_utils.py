import mlflow
import mlflow.sklearn
import os
import json
from typing import Dict,Any , Optional
from datetime import datetime
import logging
logging.basicConfig(level=INFO,format="%(asctime)s - %(levelname)s - %(message)s")

from config import get_mlflow_config

class MLflowTracker:
    def __init__(self):
        self.config = get_mlflow_config()
        self.setup_ml_flow()

    def setup_ml_flow(self):
        """Initalize MlFlow tracking with configuration"""
        tracking_uri= self.config.get('tracking_uri','file:./mlruns')
        mlflow.set_tracking_uri(tracking_uri)

        experiment_name = self.config.get('experiment_name','Churn_analysis')

        try:
            experiment_name= mlflow.get_experiment_by_name(experiment_name)
            if experiment_name is None:
                experiment_id= mlflow.create_experiment(experiment_name)
                logging.info(f"Experiment {experiment_name} created successfully with id {experiment_id}")
            else:
                experiment_id = experiment_name.experiment_id
                logging.info(f"Using existing experiment with name : {experiment_name} and id {experiment_id}")

            mlflow.set_experiment(experiment_id)
            logging.info(f"Experiment {experiment_name} set successfully")

        except Exception as e:
            logging.error(f" Error settign up MLFLOW experiment {e}")
            raise e
        
    def start_run(self,run_name:str = None,tags:Dict[str,str]=None)->mlflow.ActiveRun:
        """Starting a new MlFlow run """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if run_name is None:
            run_name = self.config.get(run_name_prefix,'run')
            run_name_prefix = run_name.replace("_"," ")
            run_name = f"{run_name_prefix} | {timestamp}"
        else:
            run_name_prefix = run_name.replace("_"," ")
            run_name = f"{run_name_prefix} | {timestamp}"

        default_tags = self.config.get('tags',{}).copy()
        if tags:
            default_tags.update(tags)
        
        run = mlflow.start_run(run_name = run_name, tags = default_tags)
        logging.info(f"Started a new MLFlow run with name {run_name} and ID {run.info.run_id}")
        return run
    
    def log_data_pipeline_metrics(self,dataset_info: Dict[str,Any]):
        """ Log data pipeline metrics and artifacts"""
        try:
            mlflow.log_metrics({
                'dataset_rows': dataset_info.get('dataset_rows',0),
                'training_rows': dataset_info.get('training_rows',0),
                'test_rows': dataset_info.get('test_rows',0),
                'number_of_features': dataset_info.get('number_of_features',0),
                'dropped_features_count': dataset_info.get('dropped_features_count',0),
                'missing_values': dataset_info.get('missing_values',0),
                'outliers_removed': dataset_info.get('outliers_removed',0)
            })

            mlflow.log_params({
                'test_size': dataset_info.get('test_size',0.2),
                'random_state': dataset_info.get('random_state',42),
                'missing_values_handled': dataset_info.get('missing_values_handled',False),
                'outliers_handled': dataset_info.get('outliers_handled',False),
                'feature_binning_handled': dataset_info.get('feature_binning_handled',False),
                'feature_encoding_handled': dataset_info.get('feature_encoding_handled',False),
                'feature_scaling_handled': dataset_info.get('feature_scaling_handled',False)
            })

            if 'feature_names' in dataset_info:
                mlflow.log_param("features", str(dataset_info.get['feature_names']))

            logging.info("Logged data pipeline to MlFlow")

        except Exception as e:
            logging.error(f"Error logging metrics in data pipeline to MlFlow. {e}")

    def log_training_metrics(self,model,model_name:str,training_metrics:Dict[str,Any],model_params: Dict[str,Any]):
        """Log training metrics, parameters and model artifacts to MlFlow"""
        try:
            #Log model parameters
            mlflow.log_params(model_params)

            #Log training metrics
            mlflow.log_metrics(training_metrics)

            artifact_path = f"{self.config.get('artifact_path','models')}/{model_name}"
            mlflow.sklearn.log_model(
                sk_model = model,
                artifact_path = artifact_path,
                registered_model_name = f"churn_prediction_{model_name}"
            )
            logging.info(f"Logged model {model_name} successfully to MlFlow")
        
        except Exception as e:
            logging.error(f"Error logging trainining metrics to MlFlow. {e}")
            raise e

        

    




        
    
    def setup_experiment(self):
