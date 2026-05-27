import io
import unicodedata
import pickle
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from fpdf import FPDF
from groq import Groq
import os
import uvicorn
import sys

# Ensure project root is on sys.path for importing src.* reliably
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.mlproject.predict_pipelines import PredictPipeline

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

# --------------------- FastAPI Setup ---------------------
app = FastAPI(title="🪀 Heart Disease Predictor & Diet Assistant")

# --------------------- Request Schemas ---------------------
class HealthProfile(BaseModel):
    # Basic information
    age: Optional[int] = None
    sex: Optional[str] = None
    gender: Optional[str] = None
    job: Optional[str] = None
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None

    # Original heart disease model features
    cp: Optional[str] = None
    trestbps: Optional[int] = None
    chol: Optional[int] = None
    fbs: Optional[str] = None
    restecg: Optional[str] = None
    thalach: Optional[int] = None
    exang: Optional[str] = None
    oldpeak: Optional[float] = None
    slope: Optional[str] = None
    ca: Optional[int] = None
    thal: Optional[str] = None

    # Wearable data
    avg_heart_rate: Optional[float] = None
    resting_heart_rate: Optional[float] = None
    sleep_hours: Optional[float] = None
    respiratory_rate: Optional[float] = None

    # Nutritional data
    calories_per_day: Optional[float] = None
    protein_g: Optional[float] = None
    carbs_g: Optional[float] = None
    fat_g: Optional[float] = None
    water_liters: Optional[float] = None

    # User-reported symptoms
    symptoms: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    language: str = "English"
def safe_index(value, options):
    if value is None or value == "I don't know":
        return None
    return options.index(value)

def yes_no_to_int(value):
    if value is None or value == "I don't know":
        return None
    return 1 if value == "Yes" else 0

def sex_to_int(value):
    if value is None or value == "I don't know":
        return None
    return 1 if value == "Male" else 0

def build_patient_context(profile: HealthProfile):
    return f"""
Additional patient information:

Basic information:
Age: {profile.age}
Biological sex: {profile.sex}
Gender: {profile.gender}
Job: {profile.job}
Weight kg: {profile.weight_kg}
Height cm: {profile.height_cm}

Wearable data:
Average heart rate: {profile.avg_heart_rate}
Resting heart rate: {profile.resting_heart_rate}
Sleep hours: {profile.sleep_hours}
Respiratory rate: {profile.respiratory_rate}

Nutrition data:
Calories per day: {profile.calories_per_day}
Protein grams per day: {profile.protein_g}
Carbs grams per day: {profile.carbs_g}
Fat grams per day: {profile.fat_g}
Water liters per day: {profile.water_liters}

User-reported symptoms:
{profile.symptoms}
"""

# --------------------- Translator ---------------------
def translate_text(text: str, target_language: str) -> str:
    if target_language == "English":
        return text
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You are a helpful translator."},
            {"role": "user", "content": f"Translate this to {target_language}:\n{text}"}
        ]
    )
    return response.choices[0].message.content.strip()

# --------------------- Endpoints ---------------------

@app.get("/debug-info")
def debug_info():
    import sys as _sys
    import sklearn as _sk
    return {"python": _sys.version, "sklearn": _sk.__version__}

@app.post("/predict")
def predict(profile: HealthProfile):
    try:
        model_input = {
            "age": profile.age,
            "sex": sex_to_int(profile.sex),
            "cp": safe_index(
                profile.cp,
                ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"]
            ),
            "trestbps": profile.trestbps,
            "chol": profile.chol,
            "fbs": yes_no_to_int(profile.fbs),
            "restecg": safe_index(
                profile.restecg,
                ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"]
            ),
            "thalach": profile.thalach,
            "exang": yes_no_to_int(profile.exang),
            "oldpeak": profile.oldpeak,
            "slope": safe_index(
                profile.slope,
                ["Upsloping", "Flat", "Downsloping"]
            ),
            "ca": profile.ca,
            "thal": safe_index(
                profile.thal,
                ["Normal", "Fixed Defect", "Reversible Defect"]
            ),
        }

        pipeline = PredictPipeline()
        prediction = pipeline.predict(model_input)

        return {
            "prediction": int(prediction),
            "risk": "High" if prediction == 1 else "Low"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/diet-plan")
def generate_diet_plan(profile: HealthProfile):
    patient_context = build_patient_context(profile)

    prompt = f"""
Create a personalized heart-healthy diet plan.

Original medical information:
Age: {profile.age}
Sex: {profile.sex}
Blood pressure: {profile.trestbps}
Cholesterol: {profile.chol}
Fasting blood sugar over 120: {profile.fbs}
Max heart rate: {profile.thalach}
ST depression: {profile.oldpeak}
Thalassemia: {profile.thal}

{patient_context}

If some information is missing, do not make up exact values.
Use only the information provided.
Include:
1. Main nutrition goals
2. Foods to eat
3. Foods to limit
4. One sample day of meals
5. Lifestyle notes based on sleep, symptoms, job, and wearable data if available
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You are a certified medical dietitian. Give educational advice, not diagnosis."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=800
    )

    return {"diet_plan": response.choices[0].message.content}


@app.post("/risk-report")
def risk_report(profile: HealthProfile, prediction: int, language: str = "English"):
    patient_context = build_patient_context(profile)

    prompt = f"""
You are a cardiology assistant. Explain why the patient was predicted as {'high' if prediction else 'low'} risk.

Original model information:
Age: {profile.age}
Sex: {profile.sex}
Cholesterol: {profile.chol}
Blood pressure: {profile.trestbps}
Max heart rate: {profile.thalach}
ST depression: {profile.oldpeak}
Exercise-induced angina: {profile.exang}
Thalassemia: {profile.thal}

{patient_context}

Important:
The ML prediction is based only on the original heart disease model fields.
The wearable, nutrition, symptom, and basic information should be used only as extra context for explanation and recommendations.
Do not claim the extra fields directly changed the ML prediction unless the model is retrained with those fields.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"risk_report": translate_text(response.choices[0].message.content.strip(), language)}


@app.post("/lifestyle")
def lifestyle_advice(profile: HealthProfile, language: str = "English"):
    patient_context = build_patient_context(profile)

    prompt = f"""
Give daily lifestyle advice for heart health.

Original medical information:
Age: {profile.age}
Sex: {profile.sex}
Blood pressure: {profile.trestbps}
Cholesterol: {profile.chol}
Max heart rate: {profile.thalach}
ST depression: {profile.oldpeak}

{patient_context}

Use wearable data, nutrition data, symptoms, and basic information if provided.
If some information is missing, say what extra information would help.
Keep the advice practical and easy to follow.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"lifestyle": translate_text(response.choices[0].message.content.strip(), language)}


@app.post("/doctor-note")
def doctor_note(profile: HealthProfile, prediction: int, language: str = "English"):
    patient_context = build_patient_context(profile)

    prompt = f"""
Draft a doctor's summary note from the patient profile and risk status.

Risk status: {"High" if prediction else "Low"}

Original medical information:
Age: {profile.age}
Sex: {profile.sex}
Blood pressure: {profile.trestbps}
Cholesterol: {profile.chol}
Max heart rate: {profile.thalach}
ST depression: {profile.oldpeak}
Exercise-induced angina: {profile.exang}
Thalassemia: {profile.thal}
Major vessels colored: {profile.ca}

{patient_context}

Write a clear summary note.
Mention missing information if important.
Do not diagnose. Recommend clinical follow-up if symptoms are concerning.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"doctor_note": translate_text(response.choices[0].message.content.strip(), language)}

@app.post("/chat")
def chatbot(request: ChatRequest):
    messages = [
        {"role": "system", "content": "You are Healthy(B), a multilingual diet and heart health expert."},
        {"role": "user", "content": request.message}
    ]
    response = client.chat.completions.create(model="openai/gpt-oss-120b", messages=messages, max_tokens=300)
    reply = response.choices[0].message.content
    return {"reply": translate_text(reply, request.language)}


if __name__ == "__main__":
    # local dev: run with `python main.py`
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
