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


# ---------------------------------------------------------------------
# Human-readable BRFSS code labels for AI-generated diet/report text
# ---------------------------------------------------------------------

SEX_LABELS = {
    1.0: "Male",
    2.0: "Female",
}

AGE_CATEGORY_LABELS = {
    1.0: "18-24",
    2.0: "25-29",
    3.0: "30-34",
    4.0: "35-39",
    5.0: "40-44",
    6.0: "45-49",
    7.0: "50-54",
    8.0: "55-59",
    9.0: "60-64",
    10.0: "65-69",
    11.0: "70-74",
    12.0: "75-79",
    13.0: "80+",
}

EDUCATION_LABELS = {
    1.0: "Never attended school or kindergarten only",
    2.0: "Grades 1-8",
    3.0: "Grades 9-11",
    4.0: "High school graduate / GED",
    5.0: "Some college or technical school",
    6.0: "College graduate",
    9.0: "Refused / unknown",
}

INCOME_LABELS = {
    1.0: "Less than $10,000",
    2.0: "$10,000 to less than $15,000",
    3.0: "$15,000 to less than $20,000",
    4.0: "$20,000 to less than $25,000",
    5.0: "$25,000 to less than $35,000",
    6.0: "$35,000 to less than $50,000",
    7.0: "$50,000 to less than $75,000",
    8.0: "$75,000 to less than $100,000",
    9.0: "$100,000 to less than $150,000",
    10.0: "$150,000 to less than $200,000",
    11.0: "$200,000 or more",
    77.0: "Don't know",
    99.0: "Refused",
}

EMPLOYMENT_LABELS = {
    1.0: "Employed for wages",
    2.0: "Self-employed",
    3.0: "Out of work for 1 year or more",
    4.0: "Out of work for less than 1 year",
    5.0: "Homemaker",
    6.0: "Student",
    7.0: "Retired",
    8.0: "Unable to work",
    9.0: "Refused",
}

MARITAL_LABELS = {
    1.0: "Married",
    2.0: "Divorced",
    3.0: "Widowed",
    4.0: "Separated",
    5.0: "Never married",
    6.0: "Unmarried couple",
    9.0: "Refused",
}

HOME_OWNERSHIP_LABELS = {
    1.0: "Own",
    2.0: "Rent",
    3.0: "Other arrangement",
    7.0: "Don't know",
    9.0: "Refused",
}

GENERAL_HEALTH_LABELS = {
    1.0: "Excellent",
    2.0: "Very good",
    3.0: "Good",
    4.0: "Fair",
    5.0: "Poor",
    7.0: "Don't know",
    9.0: "Refused",
}

YES_NO_CODE_LABELS = {
    1.0: "Yes",
    2.0: "No",
    7.0: "Don't know",
    9.0: "Refused",
}

LAST_CHECKUP_LABELS = {
    1.0: "Within past year",
    2.0: "Within past 2 years",
    3.0: "Within past 5 years",
    4.0: "5 or more years ago",
    7.0: "Don't know",
    8.0: "Never",
    9.0: "Refused",
}

SMOKER_STATUS_LABELS = {
    1.0: "Current smoker, every day",
    2.0: "Current smoker, some days",
    3.0: "Former smoker",
    4.0: "Never smoked",
    9.0: "Don't know / refused",
}

ECIG_LABELS = {
    1.0: "Use every day",
    2.0: "Use some days",
    3.0: "Not at all",
    4.0: "Never used",
    7.0: "Don't know",
    9.0: "Refused",
}

SMOKELESS_LABELS = {
    1.0: "Use every day",
    2.0: "Use some days",
    3.0: "Not at all",
    7.0: "Don't know",
    9.0: "Refused",
}


def code_to_label(value, mapping):
    if value is None:
        return "Unknown"
    try:
        return mapping.get(float(value), "Unknown")
    except (TypeError, ValueError):
        return "Unknown"


def height_code_to_label(value):
    if value is None:
        return "Unknown"
    try:
        value = int(value)
        feet = value // 100
        inches = value % 100
        return f"{feet} ft {inches} in"
    except (TypeError, ValueError):
        return "Unknown"


def alcohol_days_to_label(value):
    if value is None:
        return "Unknown"

    try:
        value = int(value)
    except (TypeError, ValueError):
        return "Unknown"

    if value == 888:
        return "No alcohol use in past 30 days"
    if 101 <= value <= 107:
        return f"{value - 100} day(s) per week"
    if 201 <= value <= 230:
        return f"{value - 200} day(s) per month"
    if value in [777, 999]:
        return "Don't know / refused"

    return f"BRFSS alcohol days code {value}"


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
    sex_label = code_to_label(profile.Sex, SEX_LABELS)
    age_label = code_to_label(profile.AgeCategory, AGE_CATEGORY_LABELS)
    education_label = code_to_label(profile.Education, EDUCATION_LABELS)
    income_label = code_to_label(profile.Income, INCOME_LABELS)
    employment_label = code_to_label(profile.EmploymentStatus, EMPLOYMENT_LABELS)
    marital_label = code_to_label(profile.MaritalStatus, MARITAL_LABELS)
    home_label = code_to_label(profile.HomeOwnership, HOME_OWNERSHIP_LABELS)
    general_health_label = code_to_label(profile.GeneralHealth, GENERAL_HEALTH_LABELS)
    good_or_better_label = code_to_label(profile.GoodOrBetterHealth, YES_NO_CODE_LABELS)
    last_checkup_label = code_to_label(profile.LastCheckup, LAST_CHECKUP_LABELS)
    height_label = height_code_to_label(profile.Height)
    smoker_status_label = code_to_label(profile.SmokerStatus, SMOKER_STATUS_LABELS)
    ecig_label = code_to_label(profile.ECigaretteUsage, ECIG_LABELS)
    smokeless_label = code_to_label(profile.SmokelessTobaccoUse, SMOKELESS_LABELS)
    alcohol_label = alcohol_days_to_label(profile.AlcoholDays)
    physical_activity_label = code_to_label(profile.PhysicalActivities, YES_NO_CODE_LABELS)

    return f"""
BRFSS prediction fields:
Sex: {sex_label} (code: {profile.Sex})
Age category: {age_label} (code: {profile.AgeCategory})
Education: {education_label} (code: {profile.Education})
Income: {income_label} (code: {profile.Income})
Employment status: {employment_label} (code: {profile.EmploymentStatus})
Marital status: {marital_label} (code: {profile.MaritalStatus})
Home ownership: {home_label} (code: {profile.HomeOwnership})
General health: {general_health_label} (code: {profile.GeneralHealth})
Good or better health: {good_or_better_label} (code: {profile.GoodOrBetterHealth})
Last checkup: {last_checkup_label} (code: {profile.LastCheckup})
Height: {height_label} (BRFSS code: {profile.Height})
Weight pounds: {profile.Weight}
Smoked 100 cigarettes: {profile.Smoked100Cigarettes}
Smoker status: {smoker_status_label} (code: {profile.SmokerStatus})
E-cigarette usage: {ecig_label} (code: {profile.ECigaretteUsage})
Smokeless tobacco use: {smokeless_label} (code: {profile.SmokelessTobaccoUse})
Alcohol days: {alcohol_label} (code: {profile.AlcoholDays})
Physical activities in past month: {physical_activity_label} (code: {profile.PhysicalActivities})
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
        "preprocessor_path": "artifacts/preprocessor.pkl",
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
- Use the readable labels provided in the patient context. Do not reinterpret numeric BRFSS codes yourself.
"""
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": "You are a careful medical dietitian. Give educational advice, not diagnosis. Use the provided readable labels instead of guessing numeric BRFSS codes."
            },
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
- Use the readable labels provided in the patient context. Do not reinterpret numeric BRFSS codes yourself.
"""
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": "You explain health model outputs carefully. Use the provided readable labels instead of guessing numeric BRFSS codes."
            },
            {"role": "user", "content": prompt}
        ],
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

Important:
- Use the readable labels provided in the patient context.
- Do not reinterpret numeric BRFSS codes yourself.
"""
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": "You give careful lifestyle advice. Use the provided readable labels instead of guessing numeric BRFSS codes."
            },
            {"role": "user", "content": prompt}
        ],
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

Important:
- Use the readable labels provided in the patient context.
- Do not reinterpret numeric BRFSS codes yourself.
"""
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": "You draft careful medical summary notes. Use the provided readable labels instead of guessing numeric BRFSS codes."
            },
            {"role": "user", "content": prompt}
        ],
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
