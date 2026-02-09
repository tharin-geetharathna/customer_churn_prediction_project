import mlflow
import mlflow.sklearn
import os
import json
from typing import Dict,Any , Optional, Union
import numpy as np
from datetime import datetime
import logging
logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(levelname)s - %(message)s")

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
            existing_experiment= mlflow.get_experiment_by_name(experiment_name)
            if existing_experiment is None:
                experiment_id= mlflow.create_experiment(experiment_name)
                logging.info(f"Experiment {experiment_name} created successfully with id {experiment_id}")
            else:
                experiment_id = existing_experiment.experiment_id
                logging.info(f"Using existing experiment with name : {experiment_name} and id {experiment_id}")

            mlflow.set_experiment(experiment_name)
            logging.info(f"Experiment {experiment_name} set successfully")

        except Exception as e:
            logging.error(f" Error settign up MLFLOW experiment {e}")
            raise e
        
    def start_run(self,run_name:str = None,tags:Dict[str,str]=None)->mlflow.ActiveRun:
        """Starting a new MlFlow run """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if run_name is None:
            run_name = self.config.get('run_name_prefix','run')
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
                mlflow.log_param("features", str(dataset_info.get('feature_names')))

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

            artifact_path = self.config.get('artifact_path','model')

            registered_model_name = f"churn_prediction_{model_name}"

            mlflow.sklearn.log_model(
                sk_model = model,
                artifact_path = artifact_path,
                registered_model_name = registered_model_name
            )
            logging.info(f"Logged model {model_name} successfully to MlFlow")
        
        except Exception as e:
            logging.error(f"Error logging trainining metrics to MlFlow. {e}")
            raise e
        
    def log_evaluation_metrics(self,evaluation_metrics: Dict[str, Any],confusion_matrix_path : Optional[str] = None):
        """Log evaluations metrics to MLflow"""
        try:
            if 'metrics' in evaluation_metrics:
                mlflow.log_metrics(evaluation_metrics.get('metrics'))
            if confusion_matrix_path and os.path.exists(confusion_matrix_path):
                mlflow.log_artifact(local_path = confusion_matrix_path, artifact_path = "evaluation")
            
            logging.info("Logged evaluation metrics to MLflow")
        except Exception as e:
            logging.error(f"Error logging evaluation metrics to MLflow. {e}")
            raise e


    def log_inference_metrics(self,predictions: np.ndarray, probabilities: Optional[np.ndarray] = None,
                                additional_data: Optional[Dict[str, Any]] = None):
        """Log inference metrics to MLflow"""
        try:
            inference_metrics = {
                "num_predictions" : len(predictions),
                'churn_count': int(np.sum(predictions)),
                'retain_count' : int(len(predictions) -  np.sum(predictions))
            }

            if probabilities is not None:
                inference_metrics.update({
                    "average_churn_probability": float(np.mean(probabilities)),
                    "high_churn_probability": int(np.sum(probabilities> 0.7)),
                    "medium_churn_probaility": int(np.sum((probabilities>0.5) & (probabilities<=0.7))),
                    "low_churn_probability": int(np.sum(probabilities<=0.5))
                })

            mlflow.log_metrics(inference_metrics)

            if additional_data:
                mlflow.log_params(additional_data)

            logging.info("Logged inference metrics to MLflow")

        except Exception as e:
            logging.error(f"Error logging inference metrics to MLflow. {e}")
            raise e
        
    def load_model_from_registry(self,model_name:str,version: Optional[str]= None, stage : Optional[str]= None):
        """ Load model from MLflow registry"""
        try:
            registered_model_name = f"churn_prediction_{model_name}"
            if version:
                model_uri = f"models:/{registered_model_name}/{version}"
            elif stage:
                model_uri = f"models:/{registered_model_name}/{stage}"
            else:
                model_uri = f"models:/{registered_model_name}/latest"
            
            model = mlflow.sklearn.load_model(model_uri)
            logging.info(f"Loaded model {registered_model_name} from MLflow registry")
            return model
        
        except Exception as e:
            logging.error(f"Error loading model {registered_model_name} from MLflow registry. {e}")
            raise e
        
    def get_latest_model_version(self,model_name):
        """ Get latest model version from MLflow registry"""
        try:
            registered_model_name = f"churn_prediction_{model_name}"
            client = mlflow.tracking.MlflowClient()
            latest_version = client.get_latest_versions(registered_model_name,stages=['None','Staging','Production'])
            if latest_version:
                return latest_version[0].version
            return None
        except Exception as e:
            logging.error(f"Error getting latest model version from MLflow registry. {e}")

    def transition_model_stage(self,model_name:str, version:Optional[str] = None,stage:str="Staging"):
        """Transition model to a specific stage"""
        try:
            registered_model_name = f"churn_prediction_{model_name}"
            if version is None:
                version = self.get_latest_model_version(registered_model_name)
            
            if version:
                client = mlflow.tracking.MlflowClient()
                client.transition_model_version_stage(
                    name =  registered_model_name,
                    version = version,
                    stage = stage
                )
                logging.info(f"Transitioned model {registered_model_name} to stage {stage}")
        except Exception as e:
            logging.error(f"Error transitioning model to stage {stage}. {e}")
            raise e
        
    def end_run(self):
        """End the current MLflow run"""
        try:
            mlflow.end_run()
            logging.info("Ended the current MLflow run")
        except Exception as e:
            logging.error(f"Error ending the MLflow run: {e}")

def setup_autolog():
    """Setup Mlflow autologging for supported frameworks """
    try:
        mlflow_config = get_mlflow_config()
        if mlflow_config.get('autolog',True):
                mlflow.sklearn.autolog()
                logging.info("Enabled autologging for scikit-learn")

    except Exception as e:
        logging.error(f"Error enabling autologging for scikit-learn. {e}")

    

def create_mlflow_run_tags(pipeline_name: str, additional_tags: Optional[Dict[str,str]]= None)-> Dict[str,str]:
    """Create MLflow run tags"""
    tags = {
        'pipeline_name': pipeline_name,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    if additional_tags:
        tags.update(additional_tags)

    return tags
    