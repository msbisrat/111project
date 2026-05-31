# src/mlproject/predict_pipeline.py

import pickle
import numpy as np
import pandas as pd

class PredictPipeline:
    def __init__(self):
        # NOTE: both files must be in the same folder — check your artifacts/ directory
        # and make sure both .pkl files exist there before running.
        with open("artifacts/model.pkl", "rb") as f:
            self.model = pickle.load(f)

        with open("artifacts/preprocessor.pkl", "rb") as f:  # fixed: was "artifact/" (missing 's')
            self.preprocessor = pickle.load(f)

    def _to_df(self, data: dict) -> pd.DataFrame:
        """Convert input dict to a single-row DataFrame, dropping None values."""
        cleaned = {k: v for k, v in data.items() if v is not None}
        return pd.DataFrame([cleaned])

    def predict(self, data: dict) -> int:
        """
        Returns the binary prediction: 1 = high risk, 0 = low risk.
        """
        df = self._to_df(data)
        transformed = self.preprocessor.transform(df)
        return int(self.model.predict(transformed)[0])

    def predict_proba(self, data: dict) -> float:
        """
        Returns the probability of heart disease (class 1) as a float 0.0–1.0.
        Used by the BRFSS risk adjuster in app.py as the base probability.

        Falls back gracefully if the model doesn't support predict_proba
        (e.g. a plain SVM), returning 0.75 for high-risk and 0.25 for low-risk.
        """
        df = self._to_df(data)
        transformed = self.preprocessor.transform(df)

        if hasattr(self.model, "predict_proba"):
            # Standard sklearn models: RandomForest, XGBoost, LogisticRegression, CatBoost
            proba = self.model.predict_proba(transformed)[0]
            # proba is [P(class=0), P(class=1)] — we want the positive class
            return float(proba[1])
        else:
            # Fallback for models without predict_proba (e.g. LinearSVC)
            # Use the binary prediction and return a fixed confidence
            prediction = int(self.model.predict(transformed)[0])
            return 0.75 if prediction == 1 else 0.25
