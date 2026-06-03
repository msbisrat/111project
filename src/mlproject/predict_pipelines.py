# src/mlproject/predict_pipelines.py

import pickle
from pathlib import Path

import pandas as pd


class PredictPipeline:
    def __init__(self):
        model_path = Path("artifacts/model.pkl")

        # Some previous versions used artifact/preprocessor.pkl,
        # but the current backend expects artifacts/preprocessor.pkl.
        preprocessor_path = Path("artifacts/preprocessor.pkl")

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        with open(preprocessor_path, "rb") as f:
            self.preprocessor = pickle.load(f)

    def _to_df(self, data: dict) -> pd.DataFrame:
        """
        Convert input dict to a single-row BRFSS DataFrame.

        The BRFSS-only model was trained with these exact feature columns,
        so prediction must always send the same columns to the preprocessor.
        If a user leaves a field blank, we fill it with a safe default value.
        """

        defaults = {
            # Demographics
            "Sex": 2.0,
            "AgeCategory": 2.0,
            "Education": 5.0,
            "Income": 7.0,
            "EmploymentStatus": 6.0,
            "MaritalStatus": 5.0,
            "HomeOwnership": 2.0,

            # General health and body measures
            "GeneralHealth": 3.0,
            "GoodOrBetterHealth": 1.0,
            "LastCheckup": 1.0,
            "Height": 504,
            "Weight": 120,

            # Health behaviors
            "Smoked100Cigarettes": "No",
            "SmokerStatus": 4.0,
            "ECigaretteUsage": 4.0,
            "SmokelessTobaccoUse": 3.0,
            "AlcoholDays": 888,
            "PhysicalActivities": 1.0,

            # Conditions
            "HadDiabetes": "No",
            "HadKidneyDisease": "No",
            "HadStroke": "No",
            "HadCOPD": "No",
            "HadDepressiveDisorder": "No",
            "HadArthritis": "No",
        }

        filled = defaults.copy()

        for key, value in data.items():
            if value is not None:
                filled[key] = value

        # Keep only the BRFSS columns the model was trained on.
        # This avoids sending extra fields like username, gender, job, symptoms, etc.
        brfss_features = [
            "Sex",
            "AgeCategory",
            "Education",
            "Income",
            "EmploymentStatus",
            "MaritalStatus",
            "HomeOwnership",
            "GeneralHealth",
            "GoodOrBetterHealth",
            "LastCheckup",
            "Height",
            "Weight",
            "Smoked100Cigarettes",
            "SmokerStatus",
            "ECigaretteUsage",
            "SmokelessTobaccoUse",
            "AlcoholDays",
            "PhysicalActivities",
            "HadDiabetes",
            "HadKidneyDisease",
            "HadStroke",
            "HadCOPD",
            "HadDepressiveDisorder",
            "HadArthritis",
        ]

        row = {col: filled[col] for col in brfss_features}
        return pd.DataFrame([row])

    def predict(self, data: dict) -> int:
        """
        Returns binary prediction:
        1 = higher heart disease risk estimate
        0 = lower heart disease risk estimate
        """
        df = self._to_df(data)
        transformed = self.preprocessor.transform(df)
        prediction = self.model.predict(transformed)[0]
        return int(prediction)

    def predict_proba(self, data: dict) -> float:
        """
        Returns probability of class 1 as a float from 0.0 to 1.0.
        Used by app.py as the base risk probability.
        """
        df = self._to_df(data)
        transformed = self.preprocessor.transform(df)

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(transformed)[0]
            return float(proba[1])

        prediction = int(self.model.predict(transformed)[0])
        return 0.75 if prediction == 1 else 0.25