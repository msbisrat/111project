"""
app.py
------
FastAPI backend for the Personalised Heart Disease Risk Chatbot.

Endpoints:
    GET  /debug-info
    POST /predict          — UCI clinical prediction + probability
    POST /diet-plan        — LLM-generated diet plan
    POST /risk-report      — LLM-generated risk explanation
    POST /lifestyle        — LLM-generated lifestyle advice
    POST /doctor-note      — LLM-generated doctor summary note
    POST /chat             — General health chatbot (profile-aware)
"""

import os
import sys
import unicodedata

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from groq import Groq
import uvicorn

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.mlproject.predict_pipelines import PredictPipeline

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI(title="🫀 Personalised Heart Disease Risk API")


# ══════════════════════════════════════════════════════════════════════════════
# REQUEST SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class HealthProfile(BaseModel):
    # ── Basic ──────────────────────────────────────────────────────────────────
    age:        Optional[int]   = None
    sex:        Optional[str]   = None   # "Male" | "Female"
    gender:     Optional[str]   = None
    job:        Optional[str]   = None
    weight_kg:  Optional[float] = None
    height_cm:  Optional[float] = None

    # ── UCI Heart Disease features ─────────────────────────────────────────────
    cp:       Optional[str]   = None
    trestbps: Optional[int]   = None
    chol:     Optional[int]   = None
    fbs:      Optional[str]   = None
    restecg:  Optional[str]   = None
    thalach:  Optional[int]   = None
    exang:    Optional[str]   = None
    oldpeak:  Optional[float] = None
    slope:    Optional[str]   = None
    ca:       Optional[int]   = None
    thal:     Optional[str]   = None

    # ── Wearable data ──────────────────────────────────────────────────────────
    avg_heart_rate:     Optional[float] = None
    resting_heart_rate: Optional[float] = None
    sleep_hours:        Optional[float] = None
    respiratory_rate:   Optional[float] = None

    # ── Nutrition ──────────────────────────────────────────────────────────────
    calories_per_day: Optional[float] = None
    protein_g:        Optional[float] = None
    carbs_g:          Optional[float] = None
    fat_g:            Optional[float] = None
    water_liters:     Optional[float] = None

    # ── Symptoms ───────────────────────────────────────────────────────────────
    symptoms: Optional[str] = None

    # ── BRFSS lifestyle features ───────────────────────────────────────────────
    # Tobacco
    smoked_100_cigarettes_label: Optional[str] = None   # "Yes" | "No"
    smoker_status_label:         Optional[str] = None   # "Every day" | "Some days" | "Not at all"
    ecig_usage_label:            Optional[str] = None
    smokeless_label:             Optional[str] = None

    # Alcohol
    alcohol_drinker_label:       Optional[str] = None   # "Yes" | "No"
    alcohol_days_per_week_display: Optional[float] = None

    # Activity
    physical_activities_label:   Optional[str] = None   # "Yes" | "No"

    # Health status
    general_health_label:        Optional[str] = None   # "Excellent" … "Poor"

    # Comorbidities
    had_angina_label:            Optional[str] = None
    had_diabetes_label:          Optional[str] = None
    had_stroke_label:            Optional[str] = None
    had_copd_label:              Optional[str] = None
    had_kidney_label:            Optional[str] = None
    had_depression_label:        Optional[str] = None
    had_arthritis_label:         Optional[str] = None

    # Demographics
    education_label:             Optional[str] = None
    income_label:                Optional[str] = None
    employment_label:            Optional[str] = None
    home_ownership_label:        Optional[str] = None
    marital_status_label:        Optional[str] = None


class ChatRequest(BaseModel):
    message:  str
    language: str = "English"


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def safe_index(value, options):
    if value is None or value == "I don't know":
        return None
    try:
        return options.index(value)
    except ValueError:
        return None

def yes_no_to_int(value):
    if value is None or value == "I don't know":
        return None
    return 1 if value == "Yes" else 0

def sex_to_int(value):
    if value is None:
        return None
    return 1 if value == "Male" else 0

def na(value, fallback="Not provided"):
    """Return value as string or fallback if None."""
    return str(value) if value is not None else fallback

def build_clinical_context(p: HealthProfile) -> str:
    """Format core UCI clinical fields for LLM prompts."""
    return f"""
Clinical information:
  Age: {na(p.age)}  |  Sex: {na(p.sex)}  |  Job: {na(p.job)}
  Weight: {na(p.weight_kg)} kg  |  Height: {na(p.height_cm)} cm
  Resting BP: {na(p.trestbps)} mmHg  |  Cholesterol: {na(p.chol)} mg/dL
  Fasting blood sugar >120: {na(p.fbs)}  |  Max HR: {na(p.thalach)}
  ST depression: {na(p.oldpeak)}  |  Exercise angina: {na(p.exang)}
  Thalassemia: {na(p.thal)}  |  Major vessels: {na(p.ca)}
  Symptoms: {na(p.symptoms)}
""".strip()

def build_lifestyle_context(p: HealthProfile) -> str:
    """Format BRFSS lifestyle fields for LLM prompts."""
    return f"""
Lifestyle & health background (BRFSS-derived):
  Smoking: {na(p.smoker_status_label)}  |  Smoked 100+ cigarettes: {na(p.smoked_100_cigarettes_label)}
  E-cigarette use: {na(p.ecig_usage_label)}  |  Smokeless tobacco: {na(p.smokeless_label)}
  Alcohol drinker: {na(p.alcohol_drinker_label)}  |  Drinks per week: {na(p.alcohol_days_per_week_display)}
  Physical activity (past 30 days): {na(p.physical_activities_label)}
  General health: {na(p.general_health_label)}
  Diagnosed conditions: angina={na(p.had_angina_label)}, diabetes={na(p.had_diabetes_label)},
    stroke={na(p.had_stroke_label)}, COPD={na(p.had_copd_label)},
    kidney disease={na(p.had_kidney_label)}, depression={na(p.had_depression_label)},
    arthritis={na(p.had_arthritis_label)}
  Education: {na(p.education_label)}  |  Income: {na(p.income_label)}
  Employment: {na(p.employment_label)}  |  Home: {na(p.home_ownership_label)}
  Marital status: {na(p.marital_status_label)}
""".strip()

def build_wearable_context(p: HealthProfile) -> str:
    """Format wearable and nutrition fields for LLM prompts."""
    return f"""
Wearable & nutrition data:
  Avg HR: {na(p.avg_heart_rate)} bpm  |  Resting HR: {na(p.resting_heart_rate)} bpm
  Sleep: {na(p.sleep_hours)} hrs/night  |  Respiratory rate: {na(p.respiratory_rate)}
  Calories: {na(p.calories_per_day)}/day  |  Protein: {na(p.protein_g)}g
  Carbs: {na(p.carbs_g)}g  |  Fat: {na(p.fat_g)}g  |  Water: {na(p.water_liters)}L
""".strip()

def translate_text(text: str, target_language: str) -> str:
    if target_language == "English":
        return text
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "system", "content": "You are a helpful translator. Translate exactly, preserving medical terminology."},
            {"role": "user",   "content": f"Translate this to {target_language}:\n{text}"}
        ]
    )
    return response.choices[0].message.content.strip()


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/debug-info")
def debug_info():
    import sklearn as _sk
    return {"python": sys.version, "sklearn": _sk.__version__}


@app.post("/predict")
def predict(profile: HealthProfile):
    """
    Stage 1 clinical prediction using the UCI-trained model.
    Returns both a binary prediction AND a probability score.
    The probability is used by the Streamlit frontend for BRFSS risk adjustment.
    """
    try:
        model_input = {
            "age":      profile.age,
            "sex":      sex_to_int(profile.sex),
            "cp":       safe_index(profile.cp,
                            ["Typical Angina","Atypical Angina","Non-anginal","Asymptomatic"]),
            "trestbps": profile.trestbps,
            "chol":     profile.chol,
            "fbs":      yes_no_to_int(profile.fbs),
            "restecg":  safe_index(profile.restecg,
                            ["Normal","ST-T Abnormality","Left Ventricular Hypertrophy"]),
            "thalach":  profile.thalach,
            "exang":    yes_no_to_int(profile.exang),
            "oldpeak":  profile.oldpeak,
            "slope":    safe_index(profile.slope,
                            ["Upsloping","Flat","Downsloping"]),
            "ca":       profile.ca,
            "thal":     safe_index(profile.thal,
                            ["Normal","Fixed Defect","Reversible Defect"]),
        }

        pipeline    = PredictPipeline()
        prediction  = pipeline.predict(model_input)      # 0 or 1
        probability = pipeline.predict_proba(model_input) # float 0.0–1.0

        return {
            "prediction":  int(prediction),
            "probability": round(float(probability), 4),
            "risk":        "High" if prediction == 1 else "Low",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/diet-plan")
def generate_diet_plan(profile: HealthProfile):
    clinical  = build_clinical_context(profile)
    lifestyle = build_lifestyle_context(profile)
    wearable  = build_wearable_context(profile)

    prompt = f"""
You are a certified medical dietitian. Create a personalised heart-healthy diet plan.

{clinical}

{lifestyle}

{wearable}

Instructions:
- Use ALL provided information to personalise the plan.
- If a field says "Not provided", do not fabricate a value — skip or note it.
- Include:
  1. Main nutrition goals based on this patient's specific risk factors
  2. Foods to eat (with reasons tied to their conditions)
  3. Foods to limit or avoid (with reasons)
  4. One sample day of meals (breakfast, lunch, dinner, snacks)
  5. Lifestyle notes based on smoking status, activity level, sleep, and symptoms if available
- End with a reminder to consult a registered dietitian or doctor before making major dietary changes.
"""
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "system", "content": "You are a certified medical dietitian. Give educational advice, not diagnosis."},
            {"role": "user",   "content": prompt}
        ],
        max_tokens=1000
    )
    return {"diet_plan": response.choices[0].message.content}


@app.post("/risk-report")
def risk_report(profile: HealthProfile, prediction: int, language: str = "English"):
    clinical  = build_clinical_context(profile)
    lifestyle = build_lifestyle_context(profile)

    prompt = f"""
You are a cardiology assistant. Explain why this patient was predicted as {'HIGH' if prediction else 'LOW'} risk for heart disease.

{clinical}

{lifestyle}

Important instructions:
- The ML prediction is based ONLY on the UCI clinical features (age, sex, BP, cholesterol, ECG, etc.).
- The BRFSS lifestyle features (smoking, activity, comorbidities, etc.) were used to compute a
  LIFESTYLE-ADJUSTED risk score on top of the clinical prediction — mention this distinction clearly.
- Do NOT claim that lifestyle fields directly changed the ML model output.
- Explain which clinical values are most concerning (or reassuring).
- Explain which lifestyle factors are raising or lowering the adjusted risk score.
- Keep the language clear and accessible to a non-medical reader.
- End with a recommendation to consult a healthcare professional.
"""
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800
    )
    return {"risk_report": translate_text(response.choices[0].message.content.strip(), language)}


@app.post("/lifestyle")
def lifestyle_advice(profile: HealthProfile, language: str = "English"):
    clinical  = build_clinical_context(profile)
    lifestyle = build_lifestyle_context(profile)
    wearable  = build_wearable_context(profile)

    prompt = f"""
You are a preventive cardiologist. Give practical daily lifestyle advice for this patient.

{clinical}

{lifestyle}

{wearable}

Instructions:
- Tailor advice to their SPECIFIC conditions and lifestyle factors.
- Cover: physical activity, smoking cessation (if applicable), alcohol, sleep, stress, diet quality.
- Mention their comorbidities (diabetes, COPD, depression, etc.) where relevant.
- If physical activity is "No", give a safe beginner exercise plan.
- If they smoke, include a cessation recommendation.
- Keep advice practical, specific, and easy to follow.
- Note any fields that were missing that would help give better advice.
"""
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800
    )
    return {"lifestyle": translate_text(response.choices[0].message.content.strip(), language)}


@app.post("/doctor-note")
def doctor_note(profile: HealthProfile, prediction: int, language: str = "English"):
    clinical  = build_clinical_context(profile)
    lifestyle = build_lifestyle_context(profile)

    prompt = f"""
Draft a professional doctor's summary note for this patient.

Risk status: {"HIGH RISK" if prediction else "LOW RISK"} (ML model prediction)

{clinical}

{lifestyle}

Format:
- Patient summary (age, sex, key clinical findings)
- Key risk factors identified (clinical and lifestyle)
- Protective factors (if any)
- Notable comorbidities
- Missing information that would improve assessment
- Recommended follow-up actions
- Standard disclaimer: this is a screening tool, not a clinical diagnosis

Write in professional medical note style. Be concise and factual.
"""
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800
    )
    return {"doctor_note": translate_text(response.choices[0].message.content.strip(), language)}


@app.post("/chat")
def chatbot(request: ChatRequest):
    """
    General health chatbot.
    The Streamlit frontend injects the user's profile summary and latest
    risk result into request.message as context before the user's question.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are Healthy(B), a knowledgeable and empathetic heart health and diet assistant. "
                "You speak clearly to non-medical users. "
                "You never diagnose — you educate and recommend professional consultation. "
                "If patient context is provided at the start of the message, use it to personalise your response."
            )
        },
        {"role": "user", "content": request.message}
    ]
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=messages,
        max_tokens=400
    )
    reply = response.choices[0].message.content
    return {"reply": translate_text(reply, request.language)}


# ── Local dev entry point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))