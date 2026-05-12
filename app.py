import io
import unicodedata
import pickle
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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
    age: int
    sex: str   # "Male" or "Female"
    cp: str
    trestbps: int
    chol: int
    fbs: str
    restecg: str
    thalach: int
    exang: str
    oldpeak: float
    slope: str
    ca: int
    thal: str

class ChatRequest(BaseModel):
    message: str
    language: str = "English"

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
            "sex": 1 if profile.sex == "Male" else 0,
            "cp": ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"].index(profile.cp),
            "trestbps": profile.trestbps,
            "chol": profile.chol,
            "fbs": 1 if profile.fbs == "Yes" else 0,
            "restecg": ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"].index(profile.restecg),
            "thalach": profile.thalach,
            "exang": 1 if profile.exang == "Yes" else 0,
            "oldpeak": profile.oldpeak,
            "slope": ["Upsloping", "Flat", "Downsloping"].index(profile.slope),
            "ca": profile.ca,
            "thal": ["Normal", "Fixed Defect", "Reversible Defect"].index(profile.thal),
        }
        pipeline = PredictPipeline()
        prediction = pipeline.predict(model_input)
        return {"prediction": int(prediction), "risk": "High" if prediction == 1 else "Low"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/diet-plan")
def generate_diet_plan(profile: HealthProfile):
    prompt = f"""
I’m a {profile.age}-year-old {profile.sex.lower()} with:
BP: {profile.trestbps}, Cholesterol: {profile.chol}, Fasting Sugar: {profile.fbs}
Max HR: {profile.thalach}, ST Depression: {profile.oldpeak}, Thalassemia: {profile.thal}
Create a heart-healthy diet plan including nutrients, foods to eat/avoid, and sample meals.
"""
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "system", "content": "You are a certified medical dietitian."},
                  {"role": "user", "content": prompt}],
        max_tokens=800
    )
    return {"diet_plan": response.choices[0].message.content}


@app.post("/risk-report")
def risk_report(profile: HealthProfile, prediction: int, language: str = "English"):
    prompt = f"""
You are a cardiologist. Explain why the patient was predicted {'high' if prediction else 'low'} risk.
Age: {profile.age}, Sex: {profile.sex}, Chol: {profile.chol}, BP: {profile.trestbps}, 
HR: {profile.thalach}, ST Depression: {profile.oldpeak}, Angina: {profile.exang}, Thal: {profile.thal}
"""
    response = client.chat.completions.create(model="openai/gpt-oss-120b", messages=[{"role": "user", "content": prompt}])
    return {"risk_report": translate_text(response.choices[0].message.content.strip(), language)}


@app.post("/lifestyle")
def lifestyle_advice(profile: HealthProfile, language: str = "English"):
    prompt = f"""
Give daily lifestyle advice on diet, exercise, stress, and sleep for a patient with:
Age: {profile.age}, Sex: {profile.sex}, BP: {profile.trestbps}, Chol: {profile.chol}, HR: {profile.thalach}, ST Depression: {profile.oldpeak}
"""
    response = client.chat.completions.create(model="openai/gpt-oss-120b", messages=[{"role": "user", "content": prompt}])
    return {"lifestyle": translate_text(response.choices[0].message.content.strip(), language)}


@app.post("/doctor-note")
def doctor_note(profile: HealthProfile, prediction: int, language: str = "English"):
    prompt = f"""
Draft a doctor's summary note from patient profile and risk status:
Age: {profile.age}, Sex: {profile.sex}, Risk: {"High" if prediction else "Low"},
BP: {profile.trestbps}, Chol: {profile.chol}, HR: {profile.thalach}, ST Depression: {profile.oldpeak},
Angina: {profile.exang}, Thalassemia: {profile.thal}, Vessels: {profile.ca}
"""
    response = client.chat.completions.create(model="openai/gpt-oss-120b", messages=[{"role": "user", "content": prompt}])
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
