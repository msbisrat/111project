"""
app.py
------
FastAPI backend for the Personalised Heart Disease Risk Chatbot.
Built on the BRFSS-only model with LLM-powered recommendations.
"""

import os
import sys
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.mlproject.predict_pipelines import PredictPipeline

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

app = FastAPI(title="BRFSS Heart Disease Predictor")

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class HealthProfile(BaseModel):
    # ── BRFSS prediction features ──────────────────────────────────────────────
    Sex:                  Optional[float] = None
    AgeCategory:          Optional[float] = None
    Education:            Optional[float] = None
    Income:               Optional[float] = None
    EmploymentStatus:     Optional[float] = None
    MaritalStatus:        Optional[float] = None
    HomeOwnership:        Optional[float] = None
    GeneralHealth:        Optional[float] = None
    GoodOrBetterHealth:   Optional[float] = None
    LastCheckup:          Optional[float] = None
    Height:               Optional[float] = None
    Weight:               Optional[float] = None
    Smoked100Cigarettes:  Optional[str]   = None
    SmokerStatus:         Optional[float] = None
    ECigaretteUsage:      Optional[float] = None
    SmokelessTobaccoUse:  Optional[float] = None
    AlcoholDays:          Optional[float] = None
    PhysicalActivities:   Optional[float] = None
    HadAngina:            Optional[str]   = None   # added — strong predictor
    HadDiabetes:          Optional[str]   = None
    HadKidneyDisease:     Optional[str]   = None
    HadStroke:            Optional[str]   = None
    HadCOPD:              Optional[str]   = None
    HadDepressiveDisorder:Optional[str]   = None
    HadArthritis:         Optional[str]   = None

    # ── Extra context for AI recommendations only ──────────────────────────────
    username:          Optional[str]   = None
    gender:            Optional[str]   = None
    job:               Optional[str]   = None
    height_cm:         Optional[float] = None
    weight_kg:         Optional[float] = None
    avg_heart_rate:    Optional[float] = None
    resting_heart_rate:Optional[float] = None
    sleep_hours:       Optional[float] = None
    respiratory_rate:  Optional[float] = None
    calories_per_day:  Optional[float] = None
    protein_g:         Optional[float] = None
    carbs_g:           Optional[float] = None
    fat_g:             Optional[float] = None
    water_liters:      Optional[float] = None
    symptoms:          Optional[str]   = None


class ChatRequest(BaseModel):
    message:  str
    language: str = "English"


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def require_groq_client():
    if client is None:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is missing. Please add it to your .env file."
        )
    return client

def na(value, fallback="Not provided"):
    return str(value) if value is not None else fallback

def translate_text(text: str, target_language: str) -> str:
    if target_language == "English":
        return text
    groq_client = require_groq_client()
    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful translator. Translate exactly, preserving medical terminology."},
            {"role": "user",   "content": f"Translate this to {target_language}:\n{text}"}
        ],
    )
    return response.choices[0].message.content.strip()

def build_brfss_context(p: HealthProfile) -> str:
    """BRFSS prediction fields — used for both prediction and LLM context."""
    return f"""
BRFSS lifestyle and health profile:
  Sex: {na(p.Sex)}  |  Age category: {na(p.AgeCategory)}
  Education: {na(p.Education)}  |  Income: {na(p.Income)}
  Employment: {na(p.EmploymentStatus)}  |  Marital status: {na(p.MaritalStatus)}
  Home ownership: {na(p.HomeOwnership)}
  General health: {na(p.GeneralHealth)}  |  Good or better health: {na(p.GoodOrBetterHealth)}
  Last checkup: {na(p.LastCheckup)}
  Height (BRFSS code): {na(p.Height)}  |  Weight (lbs): {na(p.Weight)}
  Smoked 100+ cigarettes: {na(p.Smoked100Cigarettes)}  |  Smoker status: {na(p.SmokerStatus)}
  E-cigarette use: {na(p.ECigaretteUsage)}  |  Smokeless tobacco: {na(p.SmokelessTobaccoUse)}
  Alcohol days: {na(p.AlcoholDays)}  |  Physical activities: {na(p.PhysicalActivities)}
  Angina: {na(p.HadAngina)}  |  Diabetes: {na(p.HadDiabetes)}
  Kidney disease: {na(p.HadKidneyDisease)}  |  Stroke: {na(p.HadStroke)}
  COPD: {na(p.HadCOPD)}  |  Depressive disorder: {na(p.HadDepressiveDisorder)}
  Arthritis: {na(p.HadArthritis)}
""".strip()

def build_wearable_context(p: HealthProfile) -> str:
    """Wearable and nutrition fields — extra context for LLM only."""
    return f"""
Wearable and nutrition data:
  Avg HR: {na(p.avg_heart_rate)} bpm  |  Resting HR: {na(p.resting_heart_rate)} bpm
  Sleep: {na(p.sleep_hours)} hrs/night  |  Respiratory rate: {na(p.respiratory_rate)}
  Calories: {na(p.calories_per_day)}/day  |  Protein: {na(p.protein_g)}g
  Carbs: {na(p.carbs_g)}g  |  Fat: {na(p.fat_g)}g  |  Water: {na(p.water_liters)}L
  Height cm: {na(p.height_cm)}  |  Weight kg: {na(p.weight_kg)}
  Job: {na(p.job)}  |  Symptoms: {na(p.symptoms)}
""".strip()


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/debug-info")
def debug_info():
    import sklearn as _sk
    return {
        "python":           sys.version,
        "sklearn":          _sk.__version__,
        "model":            "BRFSS-only model",
        "model_path":       "artifact/model.pkl",
        "preprocessor_path":"artifact/preprocessor.pkl",
    }


@app.post("/predict")
def predict(profile: HealthProfile):
    """
    BRFSS-only clinical prediction.
    Returns binary prediction AND probability for the risk adjuster.
    """
    try:
        model_input = {
            "Sex":                   profile.Sex,
            "AgeCategory":           profile.AgeCategory,
            "Education":             profile.Education,
            "Income":                profile.Income,
            "EmploymentStatus":      profile.EmploymentStatus,
            "MaritalStatus":         profile.MaritalStatus,
            "HomeOwnership":         profile.HomeOwnership,
            "GeneralHealth":         profile.GeneralHealth,
            "GoodOrBetterHealth":    profile.GoodOrBetterHealth,
            "LastCheckup":           profile.LastCheckup,
            "Height":                profile.Height,
            "Weight":                profile.Weight,
            "Smoked100Cigarettes":   profile.Smoked100Cigarettes,
            "SmokerStatus":          profile.SmokerStatus,
            "ECigaretteUsage":       profile.ECigaretteUsage,
            "SmokelessTobaccoUse":   profile.SmokelessTobaccoUse,
            "AlcoholDays":           profile.AlcoholDays,
            "PhysicalActivities":    profile.PhysicalActivities,
            "HadAngina":             profile.HadAngina,
            "HadDiabetes":           profile.HadDiabetes,
            "HadKidneyDisease":      profile.HadKidneyDisease,
            "HadStroke":             profile.HadStroke,
            "HadCOPD":               profile.HadCOPD,
            "HadDepressiveDisorder": profile.HadDepressiveDisorder,
            "HadArthritis":          profile.HadArthritis,
        }

        pipeline    = PredictPipeline()
        prediction  = pipeline.predict(model_input)
        probability = pipeline.predict_proba(model_input)

        return {
            "prediction":  int(prediction),
            "probability": round(float(probability), 4),
            "risk":        "High" if int(prediction) == 1 else "Low",
            "message":     "BRFSS-only model estimate, not a medical diagnosis."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/diet-plan")
def generate_diet_plan(profile: HealthProfile):
    groq_client = require_groq_client()
    brfss   = build_brfss_context(profile)
    wearable = build_wearable_context(profile)

    prompt = f"""
You are a certified medical dietitian. Create a personalised heart-healthy diet plan.

{brfss}

{wearable}

Instructions:
- Tailor advice to their specific conditions (smoking, diabetes, COPD, depression, etc.)
- If a field says "Not provided", do not fabricate a value.
- Include:
  1. Main nutrition goals based on their risk factors
  2. Foods to eat (with reasons)
  3. Foods to limit (with reasons)
  4. One sample day of meals
  5. Lifestyle notes based on activity, sleep, and symptoms if available
- End with a reminder to consult a registered dietitian or doctor.
"""
    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a certified medical dietitian. Give educational advice, not diagnosis."},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=1000,
    )
    return {"diet_plan": response.choices[0].message.content}


@app.post("/risk-report")
def risk_report(profile: HealthProfile, prediction: int, language: str = "English"):
    groq_client = require_groq_client()
    brfss = build_brfss_context(profile)

    prompt = f"""
You are a cardiology assistant. Explain this heart disease risk result in clear, patient-friendly language.

Model prediction: {"HIGH RISK" if prediction else "LOW RISK"}

{brfss}

Instructions:
- This is a BRFSS-only model estimate, not a medical diagnosis.
- Explain which specific factors (smoking, conditions, activity, age, etc.) are most influencing the result.
- Distinguish between modifiable factors (smoking, activity) and non-modifiable ones (age, prior conditions).
- Note any missing fields that would improve reliability.
- End with a recommendation to consult a healthcare professional.
"""
    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
    )
    return {"risk_report": translate_text(response.choices[0].message.content.strip(), language)}


@app.post("/lifestyle")
def lifestyle_advice(profile: HealthProfile, language: str = "English"):
    groq_client = require_groq_client()
    brfss    = build_brfss_context(profile)
    wearable = build_wearable_context(profile)

    prompt = f"""
You are a preventive cardiologist. Give practical daily lifestyle advice for this patient.

{brfss}

{wearable}

Instructions:
- Tailor advice specifically to their conditions and lifestyle factors.
- Cover: physical activity, smoking cessation (if applicable), alcohol, sleep, stress, diet.
- If physical activity is "No" or 2.0, suggest a beginner-friendly exercise plan.
- If they smoke, include a cessation recommendation.
- Mention comorbidities (diabetes, COPD, depression, arthritis) where relevant.
- Keep advice practical and easy to follow.
"""
    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
    )
    return {"lifestyle": translate_text(response.choices[0].message.content.strip(), language)}


@app.post("/doctor-note")
def doctor_note(profile: HealthProfile, prediction: int, language: str = "English"):
    groq_client = require_groq_client()
    brfss = build_brfss_context(profile)

    prompt = f"""
Draft a professional doctor's summary note for this patient.

Risk status: {"HIGH RISK" if prediction else "LOW RISK"} (BRFSS model estimate)

{brfss}

Format:
- Patient summary (age category, sex, key health indicators)
- Key risk factors identified
- Protective factors (if any)
- Notable comorbidities
- Missing information that would improve assessment
- Recommended follow-up actions
- Disclaimer: this is a screening tool, not a clinical diagnosis

Write in professional medical note style. Be concise and factual.
"""
    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
    )
    return {"doctor_note": translate_text(response.choices[0].message.content.strip(), language)}


@app.post("/chat")
def chatbot(request: ChatRequest):
    """
    Profile-aware chatbot. Streamlit injects the user's profile summary
    and latest risk result into request.message as context.
    """
    groq_client = require_groq_client()
    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are Healthy(B), a knowledgeable and empathetic heart health and diet assistant. "
                    "You speak clearly to non-medical users. "
                    "You never diagnose — you educate and recommend professional consultation. "
                    "If patient context is provided at the start of the message, use it to personalise your response."
                )
            },
            {"role": "user", "content": request.message},
        ],
        max_tokens=400,
    )
    reply = response.choices[0].message.content
    return {"reply": translate_text(reply, request.language)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))