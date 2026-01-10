import os,sys
import logging
sys.path.append(os.path.join(os.path.dirname(__file__),'..','src'))
from model_inference import ModelInference


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def inference_pipeline(model_name:str,data:dict):
    MODEL_PATHS ={
        'random_forest':'artifacts/models/random_forest.joblib',
        'xgboost':'artifacts/models/xgboost.joblib'
    }

    if model_name not in MODEL_PATHS:
        raise ValueError(f"Model {model_name} is not found in the models directory")
    model_path = MODEL_PATHS[model_name]
    inferencer= ModelInference(model_path=model_path)
    inferencer.load_model()
    inferencer.load_preprocessors()
    prediction = inferencer.predict(data)
    logging.info(f"Inferencing completed successfully")
    return prediction

if __name__=="__main__":
    user_data = {
    "RowNumber": 1,
    "CustomerId": 100000,
    "Firstname": "Tharin",
    "Lastname": "Geetharathna",
    "CreditScore": 130,
    "Geography": "France",
    "Gender": "Male",
    "Age": 50,
    "Tenure": 3,
    "Balance": 1200000.50,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 0,
    "EstimatedSalary": 750.00
}
    
    output_rfc= inference_pipeline(model_name='random_forest',data= user_data)
    output_xgb= inference_pipeline(model_name='xgboost',data= user_data)
    print(f"Random forest predicts {output_rfc}")
    print(f"xgboost predicts {output_xgb}")



    