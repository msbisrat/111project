"""
risk_adjustment.py
------------------
Adjusts the UCI ML model's raw heart disease probability using all
available BRFSS lifestyle, demographic, and comorbidity features.

Two-stage pipeline:
  Stage 1 — UCI clinical model predicts base probability  (PredictPipeline)
  Stage 2 — BRFSS-learned multipliers adjust for lifestyle (THIS FILE)

All column names match the team's renamed BRFSS CSV exactly.

Usage:
    from src.risk_adjustment import BRFSSRiskAdjuster

    adjuster = BRFSSRiskAdjuster("data/brfss_cleaned.csv")
    adjuster.fit()   # once at startup

    result = adjuster.adjust(
        base_probability=0.62,
        profile={
            # Demographics
            "sex": "Male",
            "age_category": 6,          # 1-13 BRFSS age bands
            "education": 4,             # 1-6 BRFSS education levels
            "income": 5,                # 1-11 BRFSS income levels
            "employment_status": 1,
            "marital_status": 1,
            "home_ownership": 1,
            # Health status
            "general_health": 3,        # 1=Excellent ... 5=Poor
            "good_or_better_health": 1, # 1=Yes, 2=No
            "height": 170,              # cm
            "weight": 85,               # kg
            # Smoking & tobacco
            "smoked_100_cigarettes": 1, # 1=Yes, 2=No
            "smoker_status": 1,         # 1=Daily, 2=Some days, 3=Not at all
            "ecigarette_usage": 2,
            "smokeless_tobacco_use": 2,
            # Alcohol
            "alcohol_days": 105,        # BRFSS encoding
            "alcohol_drinkers": 1,      # 1=Yes, 2=No
            # Activity
            "physical_activities": 2,   # 1=Yes, 2=No
            # Comorbidities
            "had_angina": 2,
            "had_depressive_disorder": 1,
            "had_diabetes": 1,
            "had_kidney_disease": 2,
            "had_stroke": 2,
            "had_copd": 2,
            "had_arthritis": 1,
        }
    )
"""

import os
import pandas as pd
import numpy as np


# ── Map from profile dict keys → renamed BRFSS CSV column names ───────────────
PROFILE_TO_COL = {
    # Demographics
    "sex":                    "Sex",
    "age_category":           "AgeCategory",
    "education":              "Education",
    "income":                 "Income",
    "employment_status":      "EmploymentStatus",
    "marital_status":         "MaritalStatus",
    "home_ownership":         "HomeOwnership",
    # Health status
    "general_health":         "GeneralHealth",
    "good_or_better_health":  "GoodOrBetterHealth",
    "height":                 "Height",
    "weight":                 "Weight",
    # Smoking & tobacco
    "smoked_100_cigarettes":  "Smoked100Cigarettes",
    "smoker_status":          "SmokerStatus",
    "ecigarette_usage":       "ECigaretteUsage",
    "smokeless_tobacco_use":  "SmokelessTobaccoUse",
    # Alcohol
    "alcohol_days":           "AlcoholDays",
    "alcohol_drinkers":       "AlcoholDrinkers",
    # Activity
    "physical_activities":    "PhysicalActivities",
    # Comorbidities
    "had_angina":             "HadAngina",
    "had_depressive_disorder":"HadDepressiveDisorder",
    "had_diabetes":           "HadDiabetes",
    "had_kidney_disease":     "HadKidneyDisease",
    "had_stroke":             "HadStroke",
    "had_copd":               "HadCOPD",
    "had_arthritis":          "HadArthritis",
}

TARGET_COL = "HadHeartDisease"


class BRFSSRiskAdjuster:
    """
    Learns relative risk weights from the team's renamed BRFSS CSV and
    applies them to the UCI model's base probability via log-odds adjustment.

    Features are split into three groups:
        1. Comorbidities  — strong, non-modifiable risk factors (angina, stroke, diabetes...)
        2. Lifestyle      — modifiable behavioural factors (smoking, activity, alcohol...)
        3. Demographics   — stratifiers that shift baseline risk (age, sex, BMI...)
    """

    def __init__(self, brfss_csv_path: str):
        self.csv_path = brfss_csv_path
        self.weights  = {}
        self.fitted   = False

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _rr(target: pd.Series, condition: pd.Series) -> float:
        """Relative risk: mean(target | condition) / mean(target | ~condition)."""
        exp   = target[condition].mean()
        unexp = target[~condition].mean()
        return round(float(exp / unexp), 4) if unexp > 0 else 1.0

    @staticmethod
    def _binarize_target(s: pd.Series) -> pd.Series:
        """HadHeartDisease: 1=Yes → 1, 2=No → 0, else NaN."""
        return pd.to_numeric(s, errors="coerce").map({1: 1, 2: 0})

    @staticmethod
    def _decode_alcohol_days(s: pd.Series) -> pd.Series:
        """
        Convert BRFSS AlcoholDays encoding to drinks per month.
        101-107 = 1-7 days/week  →  multiply by 4
        201-230 = 1-30 days/month → use as-is
        """
        result = pd.Series(np.nan, index=s.index)
        wk  = s.between(101, 199)
        mo  = s.between(201, 299)
        result[wk] = (s[wk] - 100) * 4
        result[mo] = (s[mo] - 200)
        return result

    @staticmethod
    def _bmi(df: pd.DataFrame) -> pd.Series:
        """Compute BMI from Height (cm) and Weight (kg)."""
        h = pd.to_numeric(df["Height"], errors="coerce") / 100  # cm → m
        w = pd.to_numeric(df["Weight"], errors="coerce")
        return w / (h ** 2)

    # ── fit() ──────────────────────────────────────────────────────────────────

    def fit(self) -> dict:
        """
        Load BRFSS CSV and compute relative risk multipliers for every feature.
        Must be called once before .adjust().
        Returns the weights dict (also stored as self.weights).
        """
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(
                f"BRFSS CSV not found at '{self.csv_path}'.\n"
                "Update the path to point to your cleaned BRFSS file."
            )

        print(f"[BRFSSRiskAdjuster] Loading: {self.csv_path}")
        df = pd.read_csv(self.csv_path, low_memory=False)
        print(f"[BRFSSRiskAdjuster] {len(df):,} rows  |  {len(df.columns)} columns")

        if TARGET_COL not in df.columns:
            raise ValueError(
                f"Target column '{TARGET_COL}' not found.\n"
                f"First 20 columns: {list(df.columns[:20])}"
            )

        df["_cvd"] = self._binarize_target(df[TARGET_COL])
        df = df.dropna(subset=["_cvd"]).copy()
        t  = df["_cvd"]
        w  = {"baseline_cvd_rate": round(t.mean(), 4)}
        print(f"[BRFSSRiskAdjuster] Baseline CVD rate: {t.mean():.3f}  ({len(df):,} valid rows)\n")

        def log(label, rr):
            w[label] = rr
            print(f"  {label:<35} RR = {rr:.4f}")

        print("── Comorbidities ─────────────────────────────────────────────────────────")

        # HadAngina (1=Yes, 2=No) — strongest single predictor of heart disease
        if "HadAngina" in df.columns:
            c = pd.to_numeric(df["HadAngina"], errors="coerce") == 1
            log("had_angina_rr", self._rr(t, c))

        # HadStroke (1=Yes, 2=No)
        if "HadStroke" in df.columns:
            c = pd.to_numeric(df["HadStroke"], errors="coerce") == 1
            log("had_stroke_rr", self._rr(t, c))

        # HadDiabetes (1=Yes, 2=Yes pregnancy only, 3=No, 4=Pre-diabetes)
        if "HadDiabetes" in df.columns:
            c = pd.to_numeric(df["HadDiabetes"], errors="coerce") == 1
            log("had_diabetes_rr", self._rr(t, c))

        # HadKidneyDisease (1=Yes, 2=No)
        if "HadKidneyDisease" in df.columns:
            c = pd.to_numeric(df["HadKidneyDisease"], errors="coerce") == 1
            log("had_kidney_rr", self._rr(t, c))

        # HadCOPD (1=Yes, 2=No)
        if "HadCOPD" in df.columns:
            c = pd.to_numeric(df["HadCOPD"], errors="coerce") == 1
            log("had_copd_rr", self._rr(t, c))

        # HadArthritis (1=Yes, 2=No)
        if "HadArthritis" in df.columns:
            c = pd.to_numeric(df["HadArthritis"], errors="coerce") == 1
            log("had_arthritis_rr", self._rr(t, c))

        # HadDepressiveDisorder (1=Yes, 2=No)
        if "HadDepressiveDisorder" in df.columns:
            c = pd.to_numeric(df["HadDepressiveDisorder"], errors="coerce") == 1
            log("had_depression_rr", self._rr(t, c))

        print("\n── Lifestyle ─────────────────────────────────────────────────────────────")

        # Smoked100Cigarettes (1=Yes, 2=No)
        if "Smoked100Cigarettes" in df.columns:
            c = pd.to_numeric(df["Smoked100Cigarettes"], errors="coerce") == 1
            log("ever_smoked_rr", self._rr(t, c))

        # SmokerStatus (1=Every day, 2=Some days, 3=Not at all)
        if "SmokerStatus" in df.columns:
            s = pd.to_numeric(df["SmokerStatus"], errors="coerce")
            log("daily_smoker_rr",    self._rr(t, s == 1))
            log("someday_smoker_rr",  self._rr(t, s == 2))

        # ECigaretteUsage (1=Every day, 2=Some days, 3=Not at all, 4=Never smoked e-cig)
        if "ECigaretteUsage" in df.columns:
            s = pd.to_numeric(df["ECigaretteUsage"], errors="coerce")
            log("daily_ecig_rr",      self._rr(t, s == 1))

        # SmokelessTobaccoUse (1=Every day, 2=Some days, 3=Not at all)
        if "SmokelessTobaccoUse" in df.columns:
            s = pd.to_numeric(df["SmokelessTobaccoUse"], errors="coerce")
            log("daily_smokeless_rr", self._rr(t, s == 1))

        # AlcoholDrinkers (1=Yes, 2=No)
        if "AlcoholDrinkers" in df.columns:
            c = pd.to_numeric(df["AlcoholDrinkers"], errors="coerce") == 1
            log("any_alcohol_rr",     self._rr(t, c))

        # AlcoholDays → heavy drinker (≥16 days/month ≈ 4+ days/week)
        if "AlcoholDays" in df.columns:
            days = self._decode_alcohol_days(pd.to_numeric(df["AlcoholDays"], errors="coerce"))
            log("heavy_alcohol_rr",   self._rr(t, days >= 16))

        # PhysicalActivities (1=Yes, 2=No)
        if "PhysicalActivities" in df.columns:
            a = pd.to_numeric(df["PhysicalActivities"], errors="coerce")
            log("no_activity_rr",     self._rr(t, a == 2))

        print("\n── Demographics & Health Status ──────────────────────────────────────────")

        # Sex (1=Male, 2=Female in BRFSS _SEX)
        if "Sex" in df.columns:
            s = pd.to_numeric(df["Sex"], errors="coerce")
            log("male_sex_rr",        self._rr(t, s == 1))

        # AgeCategory (1=18-24 ... 13=80+) — older = higher risk
        if "AgeCategory" in df.columns:
            a = pd.to_numeric(df["AgeCategory"], errors="coerce")
            log("age_older_rr",       self._rr(t, a >= 7))   # 55+ vs under 55

        # Education (1=Never attended ... 6=College graduate) — lower ed = higher risk
        if "Education" in df.columns:
            e = pd.to_numeric(df["Education"], errors="coerce")
            log("low_education_rr",   self._rr(t, e <= 3))

        # Income (1=<$10k ... 11=$200k+) — lower income = higher risk
        if "Income" in df.columns:
            i = pd.to_numeric(df["Income"], errors="coerce")
            log("low_income_rr",      self._rr(t, i <= 4))

        # GeneralHealth (1=Excellent ... 5=Poor)
        if "GeneralHealth" in df.columns:
            g = pd.to_numeric(df["GeneralHealth"], errors="coerce")
            log("poor_health_rr",     self._rr(t, g >= 4))
            log("excellent_health_rr",self._rr(t, g == 1))

        # GoodOrBetterHealth (1=Good or better, 2=Fair or Poor)
        if "GoodOrBetterHealth" in df.columns:
            g = pd.to_numeric(df["GoodOrBetterHealth"], errors="coerce")
            log("not_good_health_rr", self._rr(t, g == 2))

        # BMI from Height + Weight
        if "Height" in df.columns and "Weight" in df.columns:
            bmi = self._bmi(df)
            log("obese_bmi_rr",       self._rr(t, bmi >= 30))
            log("underweight_bmi_rr", self._rr(t, bmi < 18.5))

        self.weights = w
        self.fitted  = True
        print(f"\n[BRFSSRiskAdjuster] ✅ Fit complete — {len(w)-1} weights learned.")
        return w

    # ── adjust() ───────────────────────────────────────────────────────────────

    def adjust(self, base_probability: float, profile: dict) -> dict:
        """
        Apply all BRFSS-learned multipliers to the UCI model's base probability.

        Parameters
        ----------
        base_probability : float 0.0–1.0 from PredictPipeline
        profile          : the full user profile dict (clinical + lifestyle sections
                           merged, or just the lifestyle section — both work)

        Returns
        -------
        dict:
            base_probability     : original UCI clinical score
            adjusted_probability : final personalised score (0.0–1.0)
            risk_level           : "Low" | "Moderate" | "High"
            contributing_factors : human-readable list for the chatbot/UI
        """
        if not self.fitted:
            raise RuntimeError("Call .fit() before .adjust()")

        p        = float(np.clip(base_probability, 0.01, 0.99))
        log_odds = np.log(p / (1 - p))
        factors  = []

        def apply(rr_key, default, label):
            """Apply a weight and append to factors list."""
            nonlocal log_odds
            rr = self.weights.get(rr_key, default)
            log_odds += np.log(rr)
            direction = "+" if rr >= 1 else "-"
            pct = abs(rr - 1) * 100
            factors.append(f"{label} ({direction}{pct:.0f}% relative risk, BRFSS-derived)")

        # ── Comorbidities (applied first — highest impact) ─────────────────────

        if profile.get("had_angina") == 1:
            apply("had_angina_rr", 3.5, "History of angina")

        if profile.get("had_stroke") == 1:
            apply("had_stroke_rr", 3.0, "History of stroke")

        if profile.get("had_diabetes") == 1:
            apply("had_diabetes_rr", 2.2, "Diabetes diagnosis")

        if profile.get("had_kidney_disease") == 1:
            apply("had_kidney_rr", 1.9, "Chronic kidney disease")

        if profile.get("had_copd") == 1:
            apply("had_copd_rr", 1.8, "COPD diagnosis")

        if profile.get("had_arthritis") == 1:
            apply("had_arthritis_rr", 1.4, "Arthritis diagnosis")

        if profile.get("had_depressive_disorder") == 1:
            apply("had_depression_rr", 1.2, "Depressive disorder")

        # ── Smoking & tobacco ──────────────────────────────────────────────────

        smoker = profile.get("smoker_status")
        if smoker == 1:
            apply("daily_smoker_rr", 1.7, "Current daily smoker")
        elif smoker == 2:
            apply("someday_smoker_rr", 1.4, "Current some-days smoker")
        elif profile.get("smoked_100_cigarettes") == 1:
            # Former smoker: carries ~40% of the ever-smoked RR
            rr      = self.weights.get("ever_smoked_rr", 1.4)
            partial = 1 + (rr - 1) * 0.4
            log_odds += np.log(partial)
            factors.append(f"Former smoker (+{(partial-1)*100:.0f}% relative risk, BRFSS-derived)")

        if profile.get("ecigarette_usage") == 1:
            apply("daily_ecig_rr", 1.3, "Daily e-cigarette use")

        if profile.get("smokeless_tobacco_use") == 1:
            apply("daily_smokeless_rr", 1.2, "Daily smokeless tobacco use")

        # ── Alcohol ────────────────────────────────────────────────────────────

        alcohol_days_raw = profile.get("alcohol_days")
        if alcohol_days_raw is not None:
            s = pd.Series([alcohol_days_raw])
            days_per_month = self._decode_alcohol_days(s).iloc[0]
            if not np.isnan(days_per_month):
                if days_per_month >= 16:
                    apply("heavy_alcohol_rr", 1.15, f"Heavy alcohol use (~{days_per_month:.0f} days/month)")
                elif 1 <= days_per_month <= 8:
                    rr = 1 / self.weights.get("any_alcohol_rr", 1.0)
                    if rr < 1:
                        log_odds += np.log(rr)
                        factors.append(f"Moderate alcohol use (-{(1-rr)*100:.0f}% relative risk, BRFSS-derived)")

        # ── Physical activity ──────────────────────────────────────────────────

        activity = profile.get("physical_activities")
        if activity == 2:   # No physical activity
            apply("no_activity_rr", 1.3, "No physical activity in past 30 days")
        elif activity == 1:
            rr = 1 / self.weights.get("no_activity_rr", 1.3)
            log_odds += np.log(rr)
            factors.append(f"Physically active (-{(1-rr)*100:.0f}% relative risk, BRFSS-derived)")

        # ── Health status ──────────────────────────────────────────────────────

        gen_health = profile.get("general_health")
        if gen_health is not None:
            if gen_health >= 4:   # Fair or Poor
                apply("poor_health_rr", 3.4, "Fair or poor self-rated health")
            elif gen_health == 1: # Excellent
                rr = self.weights.get("excellent_health_rr", 0.4)
                log_odds += np.log(rr)
                factors.append(f"Excellent self-rated health (-{(1-rr)*100:.0f}% relative risk, BRFSS-derived)")

        # ── BMI from Height + Weight ───────────────────────────────────────────

        height_cm = profile.get("height")
        weight_kg = profile.get("weight")
        if height_cm and weight_kg:
            bmi = weight_kg / ((height_cm / 100) ** 2)
            if bmi >= 30:
                apply("obese_bmi_rr", 1.25, f"Obese BMI ({bmi:.1f})")
            elif bmi < 18.5:
                apply("underweight_bmi_rr", 1.1, f"Underweight BMI ({bmi:.1f})")
            elif 18.5 <= bmi < 25:
                factors.append(f"Healthy BMI ({bmi:.1f}) — no adjustment")

        # ── Demographics ───────────────────────────────────────────────────────

        sex = profile.get("sex")
        if sex == 1:   # Male
            apply("male_sex_rr", 1.2, "Male sex")

        age = profile.get("age_category")
        if age is not None and age >= 7:   # 55+
            apply("age_older_rr", 1.6, f"Age category {age} (55+)")

        education = profile.get("education")
        if education is not None and education <= 3:
            apply("low_education_rr", 1.2, "Lower education level")

        income = profile.get("income")
        if income is not None and income <= 4:
            apply("low_income_rr", 1.3, "Lower income bracket")

        # ── Final probability ──────────────────────────────────────────────────

        adj_p = float(np.clip(1 / (1 + np.exp(-log_odds)), 0.0, 1.0))

        if adj_p < 0.35:
            risk_level = "Low"
        elif adj_p < 0.65:
            risk_level = "Moderate"
        else:
            risk_level = "High"

        return {
            "base_probability":     round(base_probability, 4),
            "adjusted_probability": round(adj_p, 4),
            "risk_level":           risk_level,
            "contributing_factors": factors or ["No significant risk factors identified"],
        }

    def get_weights(self) -> dict:
        """Return all learned BRFSS weights — for the paper's results section."""
        return self.weights


# ── Smoke test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    adjuster = BRFSSRiskAdjuster("data/brfss_cleaned.csv")

    # Simulate weights without loading CSV
    adjuster.weights = {
        "baseline_cvd_rate":    0.09,
        "had_angina_rr":        3.52,
        "had_stroke_rr":        3.10,
        "had_diabetes_rr":      2.21,
        "had_kidney_rr":        1.92,
        "had_copd_rr":          1.83,
        "had_arthritis_rr":     1.44,
        "had_depression_rr":    1.19,
        "ever_smoked_rr":       1.45,
        "daily_smoker_rr":      1.72,
        "someday_smoker_rr":    1.41,
        "daily_ecig_rr":        1.30,
        "daily_smokeless_rr":   1.20,
        "any_alcohol_rr":       0.95,
        "heavy_alcohol_rr":     1.14,
        "no_activity_rr":       1.31,
        "male_sex_rr":          1.24,
        "age_older_rr":         1.65,
        "low_education_rr":     1.18,
        "low_income_rr":        1.27,
        "poor_health_rr":       3.45,
        "excellent_health_rr":  0.38,
        "not_good_health_rr":   3.20,
        "obese_bmi_rr":         1.28,
        "underweight_bmi_rr":   1.10,
    }
    adjuster.fitted = True

    print("=== High-risk profile ===")
    r1 = adjuster.adjust(0.55, {
        "sex": 1, "age_category": 9, "income": 2, "education": 2,
        "general_health": 4, "height": 175, "weight": 105,
        "smoker_status": 1, "smoked_100_cigarettes": 1,
        "ecigarette_usage": 1, "smokeless_tobacco_use": 2,
        "alcohol_days": 120, "alcohol_drinkers": 1,
        "physical_activities": 2,
        "had_angina": 1, "had_diabetes": 1, "had_stroke": 2,
        "had_kidney_disease": 1, "had_copd": 1,
        "had_arthritis": 1, "had_depressive_disorder": 1,
    })
    print(f"Base: {r1['base_probability']:.0%} → Adjusted: {r1['adjusted_probability']:.0%} ({r1['risk_level']})")
    for f in r1["contributing_factors"]: print(f"  - {f}")

    print("\n=== Low-risk profile ===")
    r2 = adjuster.adjust(0.30, {
        "sex": 2, "age_category": 4, "income": 8, "education": 6,
        "general_health": 1, "height": 165, "weight": 60,
        "smoker_status": 3, "smoked_100_cigarettes": 2,
        "ecigarette_usage": 4, "smokeless_tobacco_use": 3,
        "alcohol_days": 202, "alcohol_drinkers": 1,
        "physical_activities": 1,
        "had_angina": 2, "had_diabetes": 3, "had_stroke": 2,
        "had_kidney_disease": 2, "had_copd": 2,
        "had_arthritis": 2, "had_depressive_disorder": 2,
    })
    print(f"Base: {r2['base_probability']:.0%} → Adjusted: {r2['adjusted_probability']:.0%} ({r2['risk_level']})")
    for f in r2["contributing_factors"]: print(f"  - {f}")