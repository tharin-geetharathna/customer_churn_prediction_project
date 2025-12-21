import os 
import groq
import logging
from abc import ABC,abstractmethod
from pydantic import BaseModel
import pandas as pd
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()



class MissingValueHandlingStrategy(ABC):
    @abstractmethod
    def handle_missing_values(self, df: pd.DataFrame)-> pd.DataFrame:
        pass

class DropMissingValues(MissingValueHandlingStrategy):
    def __init__(self,drop_columns=[]):
        self.drop_columns= drop_columns
        logging.info(f" Dropping columns {self.drop_columns}")

    def handle_missing_values(self, df):
        drop_cleaned_df= df.dropna(subset=self.drop_columns)
        n_dropped = len(df)- len(drop_cleaned_df)
        logging.info(f"Dropped {n_dropped} rows due to missing values")
        return drop_cleaned_df
    
class FillMissingValueStrategy(MissingValueHandlingStrategy):
    def __init__(self,
                 imputing_column= None,
                 method='mean',
                 fill_value=None,
                 is_customer_imputer=False,
                 custom_imputer=None
                 ):
        self.imputing_column= imputing_column
        self.method= method
        self.fill_value= fill_value
        self.is_customer_imputer= is_customer_imputer
        self.custom_imputer= custom_imputer

        


    def handle_missing_values(self, df):
        if self.is_customer_imputer:
            return self.custom_imputer.impute(df)
        df_imputed= df[self.imputing_column].fillna(df[self.imputing_column].mean())
        logging.info(f"Filled missing values with {self.imputing_column} mean")
        return df


class GenderImputer():
        def __init__(self):
            self.groq_client= groq.Groq(api_key= os.getenv("GROQ_API_KEY"))
        
        def predict_gender(self,firstname,lastname):
            prompt= f"""
                What is the most likely gender of first name {firstname} and last name{lastname}?

                Your response should consits of 'Male' or 'Female'
            """
            response= self.groq_client.chat.completions.create(
                model='llama-3.3-70b-versatile',
                message=[
                    {
                        'role':'user',
                        'content':prompt
                    }
                ]
            )

            predicted_gender = response.choices[0].message.content.strip()
            logging.info(f" Predicted gender of {firstname} {lastname} is {predicted_gender}")
            return predicted_gender
            
        def impute(self, df):
            missing_gender_index= df[df['Gender'].isnull()].index
            for index in missing_gender_index:
                firstname= df.loc[index,"Firstname"]
                lastname= df.loc[index,"Lastname"]
                gender= self.predict_gender(firstname=firstname,lastname=lastname)
                if gender:
                    df.loc[index,"Gender"]= gender
                    logging.info(f" Imputed missing gender for {firstname} {lastname} as {gender}")
                else:
                    logging.info(f" Could not impute missing gender for {firstname} {lastname}")
                return df
            


        

        
        
