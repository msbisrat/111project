# # Database ----> data -----> train_test_split 

# mysql -----> local ------> train_test_split

import os
import sys
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
import pandas as pd
from src.mlproject.utils import read_sql_data
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# os: Allows working with the file system, like creating directories.

# sys: Helps with system-specific parameters and exceptions.

# CustomException: A custom class for handling exceptions.

# logging: Used to log messages, so you can track what's happening.

# pandas: A library used for handling and analyzing data.

# read_sql_data: A utility function (not shown in the code) to read data from a database.

# train_test_split: A function from Scikit-learn that splits the data into two parts: training and testing sets.

# dataclass: A decorator to easily create classes used for storing data.

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifact' , 'train.csv')
    test_data_path:str = os.path.join('artifact' , 'test.csv')
    raw_data_path:str = os.path.join('artifact' , 'raw.csv')
    raw_data_path_2:str = os.path.join('notebook\data' , 'raw.csv')
# this indicate the path where training and testing data store after spliting

# DataIngestionConfig (A configuration class):
# This class uses @dataclass to define default paths for where the data files will be saved:

# train_data_path: Where the training data will be saved.

# test_data_path: Where the testing data will be saved.

# raw_data_path: Where the raw data (before splitting) will be saved.


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
#     __init__ is a special method in Python classes.

# It's called automatically when you create (instantiate) an object of that class.

# Its job is to initialize the object — that is, set up any variables or configurations the object needs.    
    
    
    def initiate_data_ingestion(self):
        try:
            # df = read_sql_data()
            # Use repo-relative dataset to ensure portability across machines
            from pathlib import Path
            repo_root = Path(__file__).resolve().parents[3]
            df_path = repo_root / 'notebook' / 'data' / 'cleaned_data.csv'
            df = pd.read_csv(df_path)
            ###reading the data from mysql
            logging.info("reading completed mysql database")    
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path_2), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path) , exist_ok = True)
            df.to_csv(self.ingestion_config.raw_data_path  , index = False , header = True)
            df.to_csv(self.ingestion_config.raw_data_path_2, index=False, header=True)
            train_set , test_set = train_test_split(df , test_size = 0.2 , random_state = 42) 
            
            # divide data into train and test set
            train_set.to_csv(self.ingestion_config.train_data_path  , index = False , header = True)
            test_set.to_csv(self.ingestion_config.test_data_path  , index = False , header = True)
            # save the train and test data into the path
            
            logging.info("Data Ingestion is completed")
            
            # Return the paths of the train and test data files
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e , sys)    
        
    

