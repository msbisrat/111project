"""
profile_manager.py
------------------
Manages persistent user profiles for the personalized heart disease
risk chatbot. Each profile is stored as a JSON file under profiles/.

Profile structure
-----------------
{
  "username":           str,
  "created_at":         ISO datetime str,
  "updated_at":         ISO datetime str,
  "clinical":           { UCI Heart Disease features },
  "lifestyle":          { BRFSS-aligned features + label strings },
  "prediction_history": [ { timestamp, prediction, probability, adjusted_risk } ]
}

Usage
-----
    from src.mlproject.profile_manager import ProfileManager

    pm = ProfileManager()

    # Create
    pm.save_profile("alice", clinical_data={...}, lifestyle_data={...})

    # Read
    profile = pm.load_profile("alice")

    # Partial updates
    pm.update_clinical("alice",  {"chol": 210})
    pm.update_lifestyle("alice", {"smoker_status": 3})

    # Log a prediction
    pm.add_prediction("alice", prediction=1, probability=0.68, adjusted_risk=0.74)

    # LLM-ready summary string
    print(pm.get_summary("alice"))
"""

import json
import os
from datetime import datetime
from typing import Optional

# Profiles folder sits at project_root/profiles/
PROFILES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "profiles")

class ProfileManager:
    """Create, read, update and delete user profiles stored as JSON files."""

    def __init__(self, profiles_dir: str = PROFILES_DIR):
        self.profiles_dir = os.path.abspath(profiles_dir)
        os.makedirs(self.profiles_dir, exist_ok=True)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _path(self, username: str) -> str:
        safe = username.strip().lower().replace(" ", "_")
        return os.path.join(self.profiles_dir, f"{safe}.json")

    def _default_clinical(self) -> dict:
        """Empty UCI Heart Disease clinical feature set."""
        return {
            "age":       None,
            "sex":       None,   # "Male" | "Female"
            "gender":    None,   # free-text identity (optional)
            "job":       None,   # free-text occupation (optional)
            "height_cm": None,
            "weight_kg": None,
            # Core UCI features
            "cp":        None,   # chest pain type
            "trestbps":  None,   # resting blood pressure
            "chol":      None,   # cholesterol
            "fbs":       None,   # fasting blood sugar >120 mg/dL
            "restecg":   None,   # ECG result
            "thalach":   None,   # max heart rate
            "exang":     None,   # exercise-induced angina
            "oldpeak":   None,   # ST depression
            "slope":     None,   # ST slope
            "ca":        None,   # major vessels coloured
            "thal":      None,   # thalassemia
            # Optional wearable / nutrition fields
            "avg_heart_rate":    None,
            "resting_heart_rate":None,
            "sleep_hours":       None,
            "respiratory_rate":  None,
            "calories_per_day":  None,
            "protein_g":         None,
            "carbs_g":           None,
            "fat_g":             None,
            "water_liters":      None,
            "symptoms":          None,
        }

    def _default_lifestyle(self) -> dict:
        """
        Empty BRFSS-aligned lifestyle feature set.

        Numeric fields use BRFSS codes (e.g. smoker_status: 1=daily, 2=some days, 3=not at all).
        Label fields (suffix _label) store the human-readable string for display and pre-filling.
        """
        return {
            # ── Tobacco ──────────────────────────────────────────────────────
            "smoked_100_cigarettes":       None,   # 1=Yes  2=No
            "smoked_100_cigarettes_label": None,
            "smoker_status":               None,   # 1=Daily 2=Some days 3=Not at all
            "smoker_status_label":         None,
            "ecigarette_usage":            None,   # 1=Daily 2=Some days 3=Not at all 4=Never
            "ecig_usage_label":            None,
            "smokeless_tobacco_use":       None,   # 1=Daily 2=Some days 3=Not at all
            "smokeless_label":             None,

            # ── Alcohol ───────────────────────────────────────────────────────
            "alcohol_drinkers":            None,   # 1=Yes 2=No
            "alcohol_drinker_label":       None,
            "alcohol_days":                None,   # BRFSS encoding: 101-107=days/week, 201-230=days/month

            # ── Physical activity ─────────────────────────────────────────────
            "physical_activities":         None,   # 1=Yes 2=No
            "physical_activities_label":   None,

            # ── Health status ─────────────────────────────────────────────────
            "general_health":              None,   # 1=Excellent 2=Very Good 3=Good 4=Fair 5=Poor
            "general_health_label":        None,
            "good_or_better_health":       None,   # 1=Yes 2=No (derived from general_health)

            # ── Comorbidities (all: 1=Yes 2=No) ──────────────────────────────
            "had_angina":                  None,
            "had_angina_label":            None,
            "had_diabetes":                None,   # 1=Yes 2=Pregnancy 3=No 4=Pre-diabetes
            "had_diabetes_label":          None,
            "had_stroke":                  None,
            "had_stroke_label":            None,
            "had_copd":                    None,
            "had_copd_label":              None,
            "had_kidney_disease":          None,
            "had_kidney_label":            None,
            "had_depressive_disorder":     None,
            "had_depression_label":        None,
            "had_arthritis":               None,
            "had_arthritis_label":         None,

            # ── Demographics ──────────────────────────────────────────────────
            "sex":                         None,   # 1=Male 2=Female (BRFSS _SEX)
            "age_category":                None,   # 1–13 BRFSS age bands
            "education":                   None,   # 1–6 BRFSS education levels
            "education_label":             None,
            "income":                      None,   # 1–11 BRFSS income levels
            "income_label":                None,
            "employment_status":           None,   # 1–8 BRFSS employment codes
            "employment_label":            None,
            "home_ownership":              None,   # 1=Own 2=Rent 3=Other
            "home_ownership_label":        None,
            "marital_status":              None,   # 1–6 BRFSS marital codes
            "marital_status_label":        None,

            # ── Height / weight (for BMI in risk adjuster) ────────────────────
            "height":                      None,   # cm
            "weight":                      None,   # kg
        }

    # ── Core CRUD ─────────────────────────────────────────────────────────────

    def profile_exists(self, username: str) -> bool:
        return os.path.exists(self._path(username))

    def save_profile(
        self,
        username: str,
        clinical_data:  Optional[dict] = None,
        lifestyle_data: Optional[dict] = None,
    ) -> dict:
        """
        Create or fully overwrite a profile.
        Partial dicts are fine — missing keys fall back to defaults.
        Returns the saved profile dict.
        """
        profile = {
            "username":           username.strip().lower(),
            "created_at":         datetime.now().isoformat(),
            "updated_at":         datetime.now().isoformat(),
            "clinical":           {**self._default_clinical(),  **(clinical_data  or {})},
            "lifestyle":          {**self._default_lifestyle(), **(lifestyle_data or {})},
            "prediction_history": [],
        }
        with open(self._path(username), "w") as f:
            json.dump(profile, f, indent=2)
        return profile

    def load_profile(self, username: str) -> Optional[dict]:
        """Load profile from disk. Returns None if not found."""
        path = self._path(username)
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            return json.load(f)

    def update_clinical(self, username: str, updates: dict) -> Optional[dict]:
        """
        Partially update clinical fields.
        Only keys present in `updates` are changed.
        Returns updated profile or None if user not found.
        """
        profile = self.load_profile(username)
        if profile is None:
            return None
        profile["clinical"].update(updates)
        profile["updated_at"] = datetime.now().isoformat()
        with open(self._path(username), "w") as f:
            json.dump(profile, f, indent=2)
        return profile

    def update_lifestyle(self, username: str, updates: dict) -> Optional[dict]:
        """
        Partially update lifestyle fields.
        Returns updated profile or None if user not found.
        """
        profile = self.load_profile(username)
        if profile is None:
            return None
        profile["lifestyle"].update(updates)
        profile["updated_at"] = datetime.now().isoformat()
        with open(self._path(username), "w") as f:
            json.dump(profile, f, indent=2)
        return profile

    def add_prediction(
        self,
        username:      str,
        prediction:    int,
        probability:   float,
        adjusted_risk: float,
        risk_level:    str = "",
        factors:       list = None,
    ) -> Optional[dict]:
        """
        Append a prediction record to the user's history.

        Parameters
        ----------
        prediction    : 0=low risk, 1=high risk (UCI model output)
        probability   : raw UCI model probability  (0.0–1.0)
        adjusted_risk : BRFSS lifestyle-adjusted probability (0.0–1.0)
        risk_level    : "Low" | "Moderate" | "High"
        factors       : list of contributing factor strings
        """
        profile = self.load_profile(username)
        if profile is None:
            return None
        record = {
            "timestamp":     datetime.now().isoformat(),
            "prediction":    prediction,
            "probability":   round(probability, 4),
            "adjusted_risk": round(adjusted_risk, 4),
            "risk_level":    risk_level,
            "factors":       factors or [],
        }
        profile["prediction_history"].append(record)
        profile["updated_at"] = datetime.now().isoformat()
        with open(self._path(username), "w") as f:
            json.dump(profile, f, indent=2)
        return profile

    def delete_profile(self, username: str) -> bool:
        """Delete a user's profile. Returns True if deleted, False if not found."""
        path = self._path(username)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    def get_all_usernames(self) -> list:
        """Return list of all saved usernames."""
        return [
            f.replace(".json", "")
            for f in os.listdir(self.profiles_dir)
            if f.endswith(".json")
        ]

    def get_summary(self, username: str) -> Optional[str]:
        """
        Build a concise plain-English summary of the profile.
        Injected into the LLM system prompt to make the chatbot context-aware.
        """
        profile = self.load_profile(username)
        if profile is None:
            return None

        c = profile["clinical"]
        l = profile["lifestyle"]
        history = profile["prediction_history"]
        last    = history[-1] if history else None

        # ── Clinical summary ──────────────────────────────────────────────────
        age_str    = str(c.get("age")) if c.get("age") else "unknown age"
        sex_str    = c.get("sex") or "unknown sex"
        bmi_str    = ""
        if c.get("height_cm") and c.get("weight_kg"):
            bmi = c["weight_kg"] / ((c["height_cm"] / 100) ** 2)
            bmi_str = f", BMI {bmi:.1f}"

        clinical_str = (
            f"{age_str}-year-old {sex_str}{bmi_str}. "
            f"BP: {c.get('trestbps') or 'unknown'} mmHg, "
            f"Cholesterol: {c.get('chol') or 'unknown'} mg/dL, "
            f"Max HR: {c.get('thalach') or 'unknown'}."
        )

        # ── Lifestyle summary ─────────────────────────────────────────────────
        def val(key, fallback="unknown"):
            v = l.get(key)
            return str(v) if v is not None else fallback

        smoking_str = l.get("smoker_status_label") or "unknown smoking status"
        activity_str = l.get("physical_activities_label") or "unknown activity"
        health_str   = l.get("general_health_label") or "unknown general health"
        alcohol_str  = l.get("alcohol_drinker_label") or "unknown alcohol use"

        comorbidities = []
        for field, label in [
            ("had_angina_label",      "angina"),
            ("had_diabetes_label",    "diabetes"),
            ("had_stroke_label",      "stroke"),
            ("had_copd_label",        "COPD"),
            ("had_kidney_label",      "kidney disease"),
            ("had_depression_label",  "depression"),
            ("had_arthritis_label",   "arthritis"),
        ]:
            if l.get(field) == "Yes":
                comorbidities.append(label)

        comorbidity_str = (
            f"Diagnosed conditions: {', '.join(comorbidities)}."
            if comorbidities else "No reported comorbidities."
        )

        lifestyle_str = (
            f"Smoking: {smoking_str}. "
            f"Physical activity: {activity_str}. "
            f"General health: {health_str}. "
            f"Alcohol: {alcohol_str}. "
            f"{comorbidity_str} "
            f"Education: {l.get('education_label') or 'unknown'}. "
            f"Income: {l.get('income_label') or 'unknown'}."
        )

        # ── Prediction history ────────────────────────────────────────────────
        pred_str = ""
        if last:
            pred_str = (
                f" Most recent risk assessment: {last.get('risk_level', 'unknown')} "
                f"(clinical probability {last['probability']:.0%}, "
                f"lifestyle-adjusted {last['adjusted_risk']:.0%})."
            )
            if last.get("factors"):
                top_factors = last["factors"][:3]
                pred_str += f" Top factors: {'; '.join(top_factors)}."

        return (
            f"Patient profile for '{username}': "
            f"{clinical_str} "
            f"{lifestyle_str}"
            f"{pred_str}"
        )

    def get_prediction_history(self, username: str) -> list:
        """Return the full prediction history list for a user."""
        profile = self.load_profile(username)
        if profile is None:
            return []
        return profile.get("prediction_history", [])


# ── Smoke test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pm = ProfileManager(profiles_dir="./test_profiles")

    pm.save_profile(
        "test_user",
        clinical_data={
            "age": 58, "sex": "Male", "height_cm": 175, "weight_kg": 92,
            "chol": 245, "trestbps": 135, "thalach": 148,
        },
        lifestyle_data={
            "smoker_status": 1,            "smoker_status_label": "Every day",
            "smoked_100_cigarettes": 1,    "smoked_100_cigarettes_label": "Yes",
            "physical_activities": 2,      "physical_activities_label": "No",
            "general_health": 4,           "general_health_label": "Fair",
            "alcohol_drinkers": 1,         "alcohol_drinker_label": "Yes",
            "alcohol_days": 103,
            "had_diabetes": 1,             "had_diabetes_label": "Yes",
            "had_angina": 2,               "had_angina_label": "No",
            "had_stroke": 2,               "had_stroke_label": "No",
            "had_copd": 2,                 "had_copd_label": "No",
            "had_kidney_disease": 2,       "had_kidney_label": "No",
            "had_depressive_disorder": 1,  "had_depression_label": "Yes",
            "had_arthritis": 1,            "had_arthritis_label": "Yes",
            "education": 4,                "education_label": "High school graduate / GED",
            "income": 5,                   "income_label": "$25,000–$35,000",
            "height": 175,                 "weight": 92,
        }
    )
    print("✅ Profile saved.")

    pm.add_prediction(
        "test_user",
        prediction=1, probability=0.68, adjusted_risk=0.79,
        risk_level="High",
        factors=["Current daily smoker (+72%)", "No physical activity (+31%)", "Fair health (+245%)"]
    )
    print("✅ Prediction logged.")

    print("\nLLM Summary:")
    print(pm.get_summary("test_user"))

    print("\nPrediction history:")
    for p in pm.get_prediction_history("test_user"):
        print(" ", p)

    pm.delete_profile("test_user")
    print("\n✅ Test profile deleted.")