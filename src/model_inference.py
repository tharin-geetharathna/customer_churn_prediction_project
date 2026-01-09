import logging,os,sys
from pathlib import Path
import joblib
logging.basicConfig(level=logging.INFO, format= '%(asctime)s - %(levelname)s - %(message)s')

sys.path.append(os.path.join(os.path.dirname(__file__),'..','utils'))
sys.path.append(os.path.join(os.path.dirname(__file__),'..','src'))
from feature_binning import CustomBinningStrategy
from feature_encoding import NominalEncoding,Oridnal

class ModelInference:
    def __init__(self,model_path):
        self.model_path= model_path
        self.model= None

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model is not found at the location {self.model_path}")
        self.model = joblib.load(self.model_path)
        logging.info(f"Model succesfully loaded")
        
    def preprocess_input(self,data):
        