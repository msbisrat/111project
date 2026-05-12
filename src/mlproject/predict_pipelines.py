# src/mlproject/predict_pipeline.py

import pickle
import numpy as np
import pandas as pd

class PredictPipeline:
    def __init__(self):
        with open("artifacts/model.pkl", "rb") as f:
            self.model = pickle.load(f)
        with open("artifact/preprocessor.pkl", "rb") as f:
            self.preprocessor = pickle.load(f)

    def predict(self, data: dict):
        df = pd.DataFrame([data])
    
        transformed_data = self.preprocessor.transform(df)
        prediction = self.model.predict(transformed_data)[0]
    
        return prediction
