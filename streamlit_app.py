import streamlit as st
st.set_page_config(page_title="🪀 Heart Risk & Diet AI", layout="wide")

import requests
from src.mlproject.risk_adjustment import BRFSSRiskAdjuster
from src.mlproject.profile_manager  import ProfileManager

"""
streamlit_app.py
----------------
Main UI for the Personalised Heart Disease Risk Chatbot.

Two-stage prediction pipeline:
  Stage 1 — UCI clinical model via FastAPI  (/predict endpoint in app.py)
  Stage 2 — BRFSS lifestyle adjustment      (BRFSSRiskAdjuster)

Profile persistence:
  ProfileManager saves/loads each user's clinical + lifestyle data as JSON.
  Returning users get their fields pre-filled automatically.
"""

# ------------------------- Backend URL -------------------------
API_URL = "http://127.0.0.1:8000"

st.sidebar.header("🔑 Configuration")
language = st.sidebar.selectbox(
    "🌐 Select Output Language",
    ["English", "Hindi", "Spanish", "Tamil", "Bengali"]
)

st.title("🫀 Risk Of Heart Disease Predictor & Diet Assistant")


# ------------------------- Helper Functions -------------------------
def optional_number(label, min_value=None, max_value=None, step=1.0):
    value = st.text_input(label, placeholder="Leave blank if unknown")

    if value.strip() == "":
        return None

    try:
        number = float(value)

        if min_value is not None and number < min_value:
            st.warning(f"{label} should be at least {min_value}.")
            return None

        if max_value is not None and number > max_value:
            st.warning(f"{label} should be at most {max_value}.")
            return None

        if step == 1:
            return int(number)

        return number

    except ValueError:
        st.warning(f"Please enter a valid number for {label}.")
        return None

def optional_select(label, options):
    value = st.selectbox(label, ["I don't know"] + options)

    if value == "I don't know":
        return None

    return value


# ------------------------- Session State -------------------------
for key in ["predicted", "prediction", "diet_plan_text", "risk_report", "lifestyle", "doctor_note", "chat_history", "profile"]:
    if key not in st.session_state:
        if key == "predicted":
            st.session_state[key] = False
        elif key == "chat_history":
            st.session_state[key] = []
        else:
            st.session_state[key] = None


# ------------------------- Tabs -------------------------
profile_tab, diet_tab, report_tab, lifestyle_tab, doctor_tab = st.tabs(
    ["📋 Profile", "🥗 Diet Plan", "🗾 Risk Report", "🏃 Lifestyle", "📄 Doctor's Note"]
)


# ------------------------- Profile Tab -------------------------
with profile_tab:
    st.subheader("📋 Your Health Profile")
    st.caption("Patients can leave any field blank if they do not know the answer.")

    with st.expander("🏠 Basic Information", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            age = optional_number("🎂 Age", 0, 120)
            sex = optional_select("♂️ Biological Sex", ["Male", "Female"])
            gender = st.text_input("Gender optional", placeholder="Leave blank if unknown")
            job = st.text_input("Job optional", placeholder="Example: student, office worker")

        with col2:
            weight_kg = optional_number("Weight kg optional", 20, 300, step=0.1)
            height_cm = optional_number("Height cm optional", 80, 250, step=0.1)

    with st.expander("💓 Heart / Medical Information", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            trestbps = optional_number("🩺 Resting Blood Pressure mm Hg", 60, 250)
            chol = optional_number("🧪 Cholesterol Level mg/dL", 50, 600)
            thalach = optional_number("❤️ Max Heart Rate Achieved", 40, 250)
            oldpeak = optional_number("📉 ST Depression Exercise vs Rest", 0, 10, step=0.1)

        with col2:
            cp = optional_select(
                "💓 Chest Pain Type",
                ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"]
            )
            exang = optional_select("🏃 Chest pain during exercise?", ["No", "Yes"])
            fbs = optional_select("🍬 Fasting blood sugar > 120 mg/dL?", ["No", "Yes"])
            restecg = optional_select(
                "📈 ECG Results",
                ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"]
            )
            slope = optional_select("📊 Slope of ST Segment", ["Upsloping", "Flat", "Downsloping"])
            ca = optional_select("🦠 Number of Major Vessels Colored", [0, 1, 2, 3])
            thal = optional_select("🦬 Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

    with st.expander("⌚ Wearable Data Optional", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            avg_heart_rate = optional_number("Average heart rate bpm", 30, 220, step=0.1)
            resting_heart_rate = optional_number("Resting heart rate bpm", 30, 180, step=0.1)

        with col2:
            sleep_hours = optional_number("Sleeping time hours per night", 0, 24, step=0.1)
            respiratory_rate = optional_number("Respiratory rate breaths per minute", 5, 40, step=0.1)

    with st.expander("🥗 Nutrition Data Optional", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            calories_per_day = optional_number("Calories per day", 0, 10000, step=0.1)
            protein_g = optional_number("Protein grams per day", 0, 500, step=0.1)
            carbs_g = optional_number("Carbs grams per day", 0, 1000, step=0.1)

        with col2:
            fat_g = optional_number("Fat grams per day", 0, 500, step=0.1)
            water_liters = optional_number("Water liters per day", 0, 20, step=0.1)

    with st.expander("🗣️ User-Reported Symptoms Optional", expanded=False):
        symptoms = st.text_area(
            "Describe symptoms or concerns",
            placeholder="Example: chest tightness, fatigue, shortness of breath, poor sleep..."
        )

    # Prepare request payload
    profile = {
        "age": age,
        "sex": sex,
        "gender": gender or None,
        "job": job or None,
        "weight_kg": weight_kg,
        "height_cm": height_cm,

        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,

        "avg_heart_rate": avg_heart_rate,
        "resting_heart_rate": resting_heart_rate,
        "sleep_hours": sleep_hours,
        "respiratory_rate": respiratory_rate,

        "calories_per_day": calories_per_day,
        "protein_g": protein_g,
        "carbs_g": carbs_g,
        "fat_g": fat_g,
        "water_liters": water_liters,

        "symptoms": symptoms or None,
    }

required_fields = {
    "age": age,
    "blood pressure": trestbps,
    "cholesterol": chol,
    "max heart rate": thalach,
    "ST depression": oldpeak,
    "chest pain type": cp,
    "ECG result": restecg,
    "ST slope": slope,
    "major vessels": ca,
    "thalassemia": thal,
}

missing_fields = [name for name, value in required_fields.items() if value is None]

if st.button("🚑 Predict Risk"):
    st.session_state["profile"] = profile

    if len(missing_fields) >= 5:
        st.warning(
            "Too many important medical fields are missing. "
            "This prediction may not be reliable. Please enter more information if available."
        )

    res = requests.post(f"{API_URL}/predict", json=profile)

    if res.status_code == 200:
        data = res.json()
        st.session_state["prediction"] = data["prediction"]
        st.session_state["predicted"] = True
        st.session_state["missing_fields"] = missing_fields

        # Reset old generated outputs after new prediction
        st.session_state["diet_plan_text"] = None
        st.session_state["risk_report"] = None
        st.session_state["lifestyle"] = None
        st.session_state["doctor_note"] = None
    else:
        st.error("❌ Prediction failed.")
        st.write(res.text)

    if st.session_state["predicted"]:
        st.markdown("---")
        missing_fields = st.session_state.get("missing_fields", [])

    if missing_fields:
        st.info(
            f"Note: {len(missing_fields)} important fields were missing, so this result is only an estimate."
        )

    if st.session_state["prediction"] == 1:
        st.error("⚠️ **Estimated High Risk. Please consult a healthcare professional.**")
    else:
        st.success("✅ **Estimated Low Risk. Keep maintaining your health.**")


# Use saved profile for other tabs
saved_profile = st.session_state.get("profile")


# ------------------------- Diet Plan Tab -------------------------
with diet_tab:
    if st.session_state["predicted"] and saved_profile:
        if st.button("🥗 Generate Diet Plan"):
            res = requests.post(f"{API_URL}/diet-plan", json=saved_profile)
            if res.status_code == 200:
                st.session_state["diet_plan_text"] = res.json()["diet_plan"]
            else:
                st.error("❌ Diet plan generation failed.")
                st.write(res.text)

        if st.session_state["diet_plan_text"]:
            st.markdown("### 🥗 Diet Plan")
            st.markdown(st.session_state["diet_plan_text"])
    else:
        st.info("⚠️ Please complete your profile and run prediction first.")


# ------------------------- Risk Report Tab -------------------------
with report_tab:
    if st.session_state["predicted"] and saved_profile:
        if st.button("🗾 Generate Risk Report"):
            res = requests.post(
                f"{API_URL}/risk-report",
                params={"prediction": st.session_state["prediction"], "language": language},
                json=saved_profile
            )

            if res.status_code == 200:
                st.session_state["risk_report"] = res.json()["risk_report"]
            else:
                st.error("❌ Risk report generation failed.")
                st.write(res.text)

        if st.session_state.get("risk_report"):
            st.markdown("### 🗾 Risk Report")
            st.markdown(st.session_state["risk_report"])
    else:
        st.info("⚠️ Please complete your profile and run prediction first.")


# ------------------------- Lifestyle Tab -------------------------
with lifestyle_tab:
    if st.session_state["predicted"] and saved_profile:
        if st.button("🏃 Lifestyle Suggestions"):
            res = requests.post(
                f"{API_URL}/lifestyle",
                params={"language": language},
                json=saved_profile
            )

            if res.status_code == 200:
                st.session_state["lifestyle"] = res.json()["lifestyle"]
            else:
                st.error("❌ Lifestyle generation failed.")
                st.write(res.text)

        if st.session_state.get("lifestyle"):
            st.markdown("### 🏃 Lifestyle Advice")
            st.markdown(st.session_state["lifestyle"])
    else:
        st.info("⚠️ Please complete your profile and run prediction first.")


# ------------------------- Doctor's Note Tab -------------------------
with doctor_tab:
    if st.session_state["predicted"] and saved_profile:
        if st.button("📄 Generate Doctor's Note"):
            res = requests.post(
                f"{API_URL}/doctor-note",
                params={"prediction": st.session_state["prediction"], "language": language},
                json=saved_profile
            )

            if res.status_code == 200:
                st.session_state["doctor_note"] = res.json()["doctor_note"]
            else:
                st.error("❌ Doctor note generation failed.")
                st.write(res.text)

        if st.session_state.get("doctor_note"):
            st.markdown("### 📄 Doctor's Note")
            st.markdown(st.session_state["doctor_note"])
    else:
        st.info("⚠️ Please complete your profile and run prediction first.")


# ------------------------- Sidebar Chatbot -------------------------
with st.sidebar:
    st.header("💬 Diet & Medical Chatbot")
    user_input = st.chat_input("❓ Ask anything")

    if user_input:
        res = requests.post(
            f"{API_URL}/chat",
            json={"message": user_input, "language": language}
        )

        if res.status_code == 200:
            reply = res.json()["reply"]
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
        else:
            st.error("Chatbot failed.")
            st.write(res.text)

    for msg in st.session_state.chat_history[::-1]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 🪡 Chat History")
        for msg in reversed(st.session_state.chat_history):
            st.markdown(f"**{msg['role'].capitalize()}**: {msg['content']}")