from src import logging
from src import CustomException
import sys
from src import DataIngestion
from src import DataIngestionConfig
from src import DataTransformationConfig , DataTransformation
from src import ModelTrainerConfig , ModelTrainer

if __name__ =="__main__":
    logging.info("the execution has started")
    
    try:
        data_ingestion = DataIngestion()
        # data_ingestion_config = DataIngestionConfig()
        train_data_paths , test_data_paths = data_ingestion.initiate_data_ingestion()
        # data_ingestion_config = DataIngestionConfig()
        data_transformation = DataTransformation()
        train_arr , test_arr , temp =  data_transformation.initiate_data_transformation(train_data_paths , test_data_paths)
        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr , test_arr))
        
        
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e , sys)  