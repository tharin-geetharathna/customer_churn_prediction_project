import os,logging
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,roc_auc_score

logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')

class ModelEvaluator():
    def __init__(self,model_name,model):
        self.model_name = model_name
        self.model = model
        self.evaluation_results = {}
        logging.info(f"Initalized ModelEvaluator for {self.model_name}")

    def evaluate(self,X_test,y_test):
        y_test= y_test.values.ravel()

        y_pred= self.model.predict(X_test)
        y_proba= self.model.predict_proba(X_test)
        cr = classification_report(y_test,y_pred,output_dict=True)
        cm= confusion_matrix(y_test,y_pred)
        accuracy= accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        f1 = f1_score(y_test,y_pred)
        roc_auc = roc_auc_score(y_test,y_proba[:,1])
        self.evaluation_results={
            'classification_report': cr,
            'confusion_matrix':cm.tolist(),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }
        logging.info(f"Returned evaluation results for {self.model_name}")
        return self.evaluation_results
        
    



