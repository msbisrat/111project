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


class HealthProfile(BaseModel):
    # BRFSS-only prediction features
    Sex: Optional[float] = None
    AgeCategory: Optional[float] = None
    Education: Optional[float] = None
    Income: Optional[float] = None
    EmploymentStatus: Optional[float] = None
    MaritalStatus: Optional[float] = None
    HomeOwnership: Optional[float] = None
    GeneralHealth: Optional[float] = None
    GoodOrBetterHealth: Optional[float] = None
    LastCheckup: Optional[float] = None
    Height: Optional[float] = None
    Weight: Optional[float] = None
    Smoked100Cigarettes: Optional[str] = None
    SmokerStatus: Optional[float] = None
    ECigaretteUsage: Optional[float] = None
    SmokelessTobaccoUse: Optional[float] = None
    AlcoholDays: Optional[float] = None
    PhysicalActivities: Optional[float] = None
    HadDiabetes: Optional[str] = None
    HadKidneyDisease: Optional[str] = None
    HadStroke: Optional[str] = None
    HadCOPD: Optional[str] = None
    HadDepressiveDisorder: Optional[str] = None
    HadArthritis: Optional[str] = None

    # Extra context for AI recommendations only
    username: Optional[str] = None
    gender: Optional[str] = None
    job: Optional[str] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    avg_heart_rate: Optional[float] = None
    resting_heart_rate: Optional[float] = None
    sleep_hours: Optional[float] = None
    respiratory_rate: Optional[float] = None
    calories_per_day: Optional[float] = None
    protein_g: Optional[float] = None
    carbs_g: Optional[float] = None
    fat_g: Optional[float] = None
    water_liters: Optional[float] = None
    symptoms: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    language: str = "English"


def require_groq_client():
    if client is None:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is missing. Please add it to your .env file."
        )
    return client


def translate_text(text: str, target_language: str) -> str:
    if target_language == "English":
        return text

    groq_client = require_groq_client()
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You are a helpful translator."},
            {"role": "user", "content": f"Translate this to {target_language}:\n{text}"}
        ],
    )
    return response.choices[0].message.content.strip()


def build_patient_context(profile: HealthProfile) -> str:
    return f"""
BRFSS prediction fields:
Sex: {profile.Sex}
Age category: {profile.AgeCategory}
Education: {profile.Education}
Income: {profile.Income}
Employment status: {profile.EmploymentStatus}
Marital status: {profile.MaritalStatus}
Home ownership: {profile.HomeOwnership}
General health: {profile.GeneralHealth}
Good or better health: {profile.GoodOrBetterHealth}
Last checkup: {profile.LastCheckup}
Height BRFSS code: {profile.Height}
Weight pounds: {profile.Weight}
Smoked 100 cigarettes: {profile.Smoked100Cigarettes}
Smoker status: {profile.SmokerStatus}
E-cigarette usage: {profile.ECigaretteUsage}
Smokeless tobacco use: {profile.SmokelessTobaccoUse}
Alcohol days: {profile.AlcoholDays}
Physical activities: {profile.PhysicalActivities}
Diabetes: {profile.HadDiabetes}
Kidney disease: {profile.HadKidneyDisease}
Stroke: {profile.HadStroke}
COPD: {profile.HadCOPD}
Depressive disorder: {profile.HadDepressiveDisorder}
Arthritis: {profile.HadArthritis}

Extra context for AI only:
Username: {profile.username}
Gender: {profile.gender}
Job: {profile.job}
Height cm: {profile.height_cm}
Weight kg: {profile.weight_kg}
Average heart rate: {profile.avg_heart_rate}
Resting heart rate: {profile.resting_heart_rate}
Sleep hours: {profile.sleep_hours}
Respiratory rate: {profile.respiratory_rate}
Calories per day: {profile.calories_per_day}
Protein: {profile.protein_g}
Carbs: {profile.carbs_g}
Fat: {profile.fat_g}
Water liters: {profile.water_liters}
Symptoms: {profile.symptoms}
"""


@app.get("/debug-info")
def debug_info():
    import sklearn as _sk
    return {
        "python": sys.version,
        "sklearn": _sk.__version__,
        "model": "Option B BRFSS-only model",
        "model_path": "artifacts/model.pkl",
        "preprocessor_path": "artifact/preprocessor.pkl",
    }


@app.post("/predict")
def predict(profile: HealthProfile):
    try:
        model_input = {
            "Sex": profile.Sex,
            "AgeCategory": profile.AgeCategory,
            "Education": profile.Education,
            "Income": profile.Income,
            "EmploymentStatus": profile.EmploymentStatus,
            "MaritalStatus": profile.MaritalStatus,
            "HomeOwnership": profile.HomeOwnership,
            "GeneralHealth": profile.GeneralHealth,
            "GoodOrBetterHealth": profile.GoodOrBetterHealth,
            "LastCheckup": profile.LastCheckup,
            "Height": profile.Height,
            "Weight": profile.Weight,
            "Smoked100Cigarettes": profile.Smoked100Cigarettes,
            "SmokerStatus": profile.SmokerStatus,
            "ECigaretteUsage": profile.ECigaretteUsage,
            "SmokelessTobaccoUse": profile.SmokelessTobaccoUse,
            "AlcoholDays": profile.AlcoholDays,
            "PhysicalActivities": profile.PhysicalActivities,
            "HadDiabetes": profile.HadDiabetes,
            "HadKidneyDisease": profile.HadKidneyDisease,
            "HadStroke": profile.HadStroke,
            "HadCOPD": profile.HadCOPD,
            "HadDepressiveDisorder": profile.HadDepressiveDisorder,
            "HadArthritis": profile.HadArthritis,
        }

        pipeline = PredictPipeline()
        prediction = pipeline.predict(model_input)

        return {
            "prediction": int(prediction),
            "risk": "High" if int(prediction) == 1 else "Low",
            "message": "BRFSS-only model estimate, not a medical diagnosis."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/diet-plan")
def generate_diet_plan(profile: HealthProfile):
    groq_client = require_groq_client()
    prompt = f"""
Create a personalized heart-healthy diet plan.

{build_patient_context(profile)}

Important:
- This is educational advice, not a diagnosis.
- Use wearable, nutrition, symptoms, height/weight, and job information if provided.
- Do not make up missing values.
"""
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You are a careful medical dietitian. Give educational advice, not diagnosis."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=900,
    )
    return {"diet_plan": response.choices[0].message.content}


@app.post("/risk-report")
def risk_report(profile: HealthProfile, prediction: int, language: str = "English"):
    groq_client = require_groq_client()
    prompt = f"""
Explain this BRFSS-only model risk result in patient-friendly language.

Model prediction: {"High" if prediction else "Low"}

{build_patient_context(profile)}

Important:
- This is a model estimate/prototype, not a medical diagnosis.
- The prediction is based only on the BRFSS fields, not UCI clinical fields.
- Mention missing fields if they may affect reliability.
"""
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=700,
    )
    return {"risk_report": translate_text(response.choices[0].message.content.strip(), language)}


@app.post("/lifestyle")
def lifestyle_advice(profile: HealthProfile, language: str = "English"):
    groq_client = require_groq_client()
    prompt = f"""
Give practical daily lifestyle advice for heart health.

{build_patient_context(profile)}

Use available information about general health, activity, smoking, alcohol, chronic conditions,
wearable data, sleep, nutrition, symptoms, and job if provided.
"""
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=700,
    )
    return {"lifestyle": translate_text(response.choices[0].message.content.strip(), language)}


@app.post("/doctor-note")
def doctor_note(profile: HealthProfile, prediction: int, language: str = "English"):
    groq_client = require_groq_client()
    prompt = f"""
Draft a concise doctor's summary note from the patient profile and BRFSS-only model risk status.

Model risk status: {"High" if prediction else "Low"}

{build_patient_context(profile)}

Do not diagnose. Mention missing information if important. Recommend clinical follow-up if symptoms are concerning.
"""
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=700,
    )
    return {"doctor_note": translate_text(response.choices[0].message.content.strip(), language)}


@app.post("/chat")
def chatbot(request: ChatRequest):
    groq_client = require_groq_client()
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You are Healthy(B), a multilingual diet and heart health expert."},
            {"role": "user", "content": request.message},
        ],
        max_tokens=300,
    )
    reply = response.choices[0].message.content
    return {"reply": translate_text(reply, request.language)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
