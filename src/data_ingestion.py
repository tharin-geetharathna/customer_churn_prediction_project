import os 
import logging 
import pandas as pd
from abc import ABC,abstractmethod
logging.basicConfig(level= logging.INFO, format= '%(asctime)s - %(levelname)s - %(message)s')

class DataIngestor(ABC):
    @abstractmethod
    def ingest(self,file_path:str):
        pass


class DataIngestorCSV(DataIngestor):
    def ingest(self, file_path):
        logging.info(f" Ingesting CSV data from {file_path}")
        return pd.read_csv(file_path)
    
class DataIngestorExcel(DataIngestor):
    def ingest(self,file_path):
        logging.info(f" Ingesting Excel data from {file_path}")
        return pd.read_excel(file_path)
    
