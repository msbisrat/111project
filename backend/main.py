from backend.src import logging
from backend.src import CustomException
import sys
from backend.src.mlproject.components.data_ingestion import DataIngestion
from backend.src.mlproject.components.data_ingestion import DataIngestionConfig
from backend.src import DataTransformationConfig , DataTransformation
from backend.src import ModelTrainerConfig , ModelTrainer

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