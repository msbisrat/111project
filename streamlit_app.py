import json
from pathlib import Path

import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000"
PROFILE_FILE = Path("saved_profiles.json")

st.set_page_config(page_title="BRFSS Heart Risk & Diet AI", layout="wide")

st.sidebar.header("Configuration")
language = st.sidebar.selectbox(
    "Select Output Language",
    ["English", "Hindi", "Spanish", "Tamil", "Bengali"],
)

st.title("Heart Disease Risk Predictor & Diet Assistant")
st.caption("A model that uses CDC lifestyle factors from BRFSS data to predict health status.")


def load_profiles():
    if PROFILE_FILE.exists():
        with open(PROFILE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_profiles(profiles):
    with open(PROFILE_FILE, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)


def optional_number(label, min_value=None, max_value=None, step=1.0, key=None, default=None):
    if default is None:
        default = ""
    value = st.text_input(label, value=str(default) if default != "" else "", placeholder="Leave blank if unknown", key=key)

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


def optional_select(label, options, key=None, default=None):
    labels = ["I don't know"] + list(options.keys())

    default_label = "I don't know"
    if default is not None:
        for label_text, code in options.items():
            if code == default:
                default_label = label_text
                break

    index = labels.index(default_label) if default_label in labels else 0
    value = st.selectbox(label, labels, index=index, key=key)

    if value == "I don't know":
        return None

    return options[value]


def optional_text_select(label, options, key=None, default=None):
    labels = ["I don't know"] + options
    index = labels.index(default) if default in labels else 0
    value = st.selectbox(label, labels, index=index, key=key)

    if value == "I don't know":
        return None

    return value


def brfss_height_code(feet, inches):
    if feet is None or inches is None:
        return None
    return int(feet * 100 + inches)


SEX_OPTIONS = {"Male": 1.0, "Female": 2.0}

AGE_OPTIONS = {
    "18-24": 1.0, "25-29": 2.0, "30-34": 3.0, "35-39": 4.0,
    "40-44": 5.0, "45-49": 6.0, "50-54": 7.0, "55-59": 8.0,
    "60-64": 9.0, "65-69": 10.0, "70-74": 11.0, "75-79": 12.0,
    "80+": 13.0,
}

EDUCATION_OPTIONS = {
    "Never attended school / kindergarten only": 1.0,
    "Grades 1-8": 2.0,
    "Grades 9-11": 3.0,
    "High school graduate / GED": 4.0,
    "Some college or technical school": 5.0,
    "College graduate": 6.0,
    "Decline to state": 9.0,
}

INCOME_OPTIONS = {
    "Less than $10,000": 1.0,
    "$10,000 to < $15,000": 2.0,
    "$15,000 to < $20,000": 3.0,
    "$20,000 to < $25,000": 4.0,
    "$25,000 to < $35,000": 5.0,
    "$35,000 to < $50,000": 6.0,
    "$50,000 to < $75,000": 7.0,
    "$75,000 to < $100,000": 8.0,
    "$100,000 to < $150,000": 9.0,
    "$150,000 to < $200,000": 10.0,
    "$200,000 or more": 11.0,
    "Don't know": 77.0,
    "Decline to state": 99.0,
}

EMPLOYMENT_OPTIONS = {
    "Employed for wages": 1.0,
    "Self-employed": 2.0,
    "Out of work for 1 year or more": 3.0,
    "Out of work for less than 1 year": 4.0,
    "Homemaker": 5.0,
    "Student": 6.0,
    "Retired": 7.0,
    "Unable to work": 8.0,
    "Decline to state": 9.0,
}

MARITAL_OPTIONS = {
    "Married": 1.0,
    "Divorced": 2.0,
    "Widowed": 3.0,
    "Separated": 4.0,
    "Never married": 5.0,
    "Unmarried couple": 6.0,
    "Decline to state": 9.0,
}

HOME_OPTIONS = {
    "Own": 1.0,
    "Rent": 2.0,
    "Other arrangement": 3.0,
    "Don't know": 7.0,
    "Decline to state": 9.0,
}

GENERAL_HEALTH_OPTIONS = {
    "Excellent": 1.0,
    "Very good": 2.0,
    "Good": 3.0,
    "Fair": 4.0,
    "Poor": 5.0,
    "Don't know": 7.0,
    "Decline to state": 9.0,
}

YES_NO_CODE_OPTIONS = {
    "Yes": 1.0,
    "No": 2.0,
    "Don't know": 7.0,
    "Decline to state": 9.0,
}

LAST_CHECKUP_OPTIONS = {
    "Within past year": 1.0,
    "Within past 2 years": 2.0,
    "Within past 5 years": 3.0,
    "5 or more years ago": 4.0,
    "Don't know": 7.0,
    "Never": 8.0,
    "Decline to state": 9.0,
}

SMOKER_STATUS_OPTIONS = {
    "Current smoker - every day": 1.0,
    "Current smoker - some days": 2.0,
    "Former smoker": 3.0,
    "Never smoked": 4.0,
    "Decline to state": 9.0,
}

ECIG_OPTIONS = {
    "Use every day": 1.0,
    "Use some days": 2.0,
    "Not at all": 3.0,
    "Never used": 4.0,
    "Don't know": 7.0,
    "Decline to state": 9.0,
}

SMOKELESS_OPTIONS = {
    "Use every day": 1.0,
    "Use some days": 2.0,
    "Not at all": 3.0,
    "Don't know": 7.0,
    "Decline to state": 9.0,
}

YES_NO_TEXT_OPTIONS = ["Yes", "No"]

DIABETES_OPTIONS = [
    "Yes",
    "No",
    "No, pre-diabetes or borderline diabetes",
    "Yes, only during pregnancy",
]


for key in ["predicted", "prediction", "diet_plan_text", "risk_report", "lifestyle", "doctor_note", "chat_history", "profile"]:
    if key not in st.session_state:
        if key == "predicted":
            st.session_state[key] = False
        elif key == "chat_history":
            st.session_state[key] = []
        else:
            st.session_state[key] = None


profile_tab, diet_tab, report_tab, lifestyle_tab, doctor_tab = st.tabs(
    ["Profile", "Diet Plan", "Risk Report", "Lifestyle", "Doctor's Note"]
)


with profile_tab:
    st.subheader("Patient Health Profile")

    profiles = load_profiles()

    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        username = st.text_input("Username for saving/loading profile", placeholder="Example: sarah")
    with col_b:
        load_clicked = st.button("Load Profile")
    with col_c:
        save_clicked = st.button("Save Profile")

    loaded = {}
    if load_clicked and username:
        loaded = profiles.get(username, {})

        if loaded:
            st.session_state["loaded_profile"] = loaded

            # For selectbox widgets, Streamlit session_state must store the visible label,
            # not the numeric code saved in the profile.
            selectbox_option_maps = {
                "Sex": SEX_OPTIONS,
                "AgeCategory": AGE_OPTIONS,
                "Education": EDUCATION_OPTIONS,
                "Income": INCOME_OPTIONS,
                "EmploymentStatus": EMPLOYMENT_OPTIONS,
                "MaritalStatus": MARITAL_OPTIONS,
                "HomeOwnership": HOME_OPTIONS,
                "GeneralHealth": GENERAL_HEALTH_OPTIONS,
                "LastCheckup": LAST_CHECKUP_OPTIONS,
                "SmokerStatus": SMOKER_STATUS_OPTIONS,
                "ECigaretteUsage": ECIG_OPTIONS,
                "SmokelessTobaccoUse": SMOKELESS_OPTIONS,
                "PhysicalActivities": YES_NO_CODE_OPTIONS,
            }

            text_selectbox_keys = [
                "Smoked100Cigarettes",
                "HadDiabetes",
                "HadKidneyDisease",
                "HadStroke",
                "HadCOPD",
                "HadDepressiveDisorder",
                "HadArthritis",
            ]

            for key, value in loaded.items():
                if value is None:
                    continue

                if key in selectbox_option_maps:
                    # Convert saved numeric code back to the display label.
                    for label, code in selectbox_option_maps[key].items():
                        if code == value:
                            st.session_state[key] = label
                            break
                elif key in text_selectbox_keys:
                    # Text selectboxes save strings like "Yes" or "No".
                    st.session_state[key] = value
                elif key in [
                    "height_feet", "height_inches", "Weight", "AlcoholDays",
                    "height_cm", "weight_kg", "avg_heart_rate", "resting_heart_rate",
                    "sleep_hours", "respiratory_rate", "calories_per_day",
                    "protein_g", "carbs_g", "fat_g", "water_liters",
                    "gender", "job", "symptoms"
                ]:
                    # Text input / text area widgets must store strings in session_state.
                    st.session_state[key] = str(value)

            st.success(f"Loaded profile for {username}")
            st.rerun()

        else:
            st.warning("No saved profile found for this username.")

    loaded = st.session_state.get("loaded_profile", {})

    st.markdown("### BRFSS Prediction Fields")
    st.caption("These fields are used directly by the BRFSS-only model.")

    with st.expander("Demographics", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            Sex = optional_select("Biological Sex", SEX_OPTIONS, key="Sex", default=loaded.get("Sex"))
            MaritalStatus = optional_select("Marital Status (optional)", MARITAL_OPTIONS, key="MaritalStatus", default=loaded.get("MaritalStatus"))
            EmploymentStatus = optional_select("Employment Status (optional)", EMPLOYMENT_OPTIONS, key="EmploymentStatus", default=loaded.get("EmploymentStatus"))
            HomeOwnership = optional_select("Home Ownership (optional)", HOME_OPTIONS, key="HomeOwnership", default=loaded.get("HomeOwnership"))

        with col2:
            AgeCategory = optional_select("Age Category", AGE_OPTIONS, key="AgeCategory", default=loaded.get("AgeCategory"))
            Education = optional_select("Education (optional)", EDUCATION_OPTIONS, key="Education", default=loaded.get("Education"))
            Income = optional_select("Income (optional)", INCOME_OPTIONS, key="Income", default=loaded.get("Income"))
        
    with st.expander("General Health and Body Measures", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            GeneralHealth = optional_select("How would you classify your general health?", GENERAL_HEALTH_OPTIONS, key="GeneralHealth", default=loaded.get("GeneralHealth"))
            height_feet = optional_number("What is your height, feet part; e.g. 5 if 5'6", 3, 8, key="height_feet", default=loaded.get("height_feet", ""))
            Weight = optional_number("What is your weight in pounds", 50, 700, key="Weight", default=loaded.get("Weight", ""))

        with col2:
            LastCheckup = optional_select("When was your last routine checkup?", LAST_CHECKUP_OPTIONS, key="LastCheckup", default=loaded.get("LastCheckup"))
            height_inches = optional_number("What is your height, inches part; e.g. 6 if 5'6", 0, 11, key="height_inches", default=loaded.get("height_inches", ""))
            Height = brfss_height_code(height_feet, height_inches)

    with st.expander("Health Behaviors and Conditions", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            Smoked100Cigarettes = optional_text_select("Have you smoked at least 100 cigarettes in your life?", YES_NO_TEXT_OPTIONS, key="Smoked100Cigarettes", default=loaded.get("Smoked100Cigarettes"))
            SmokerStatus = optional_select("Smoker Status", SMOKER_STATUS_OPTIONS, key="SmokerStatus", default=loaded.get("SmokerStatus"))
            ECigaretteUsage = optional_select("How often do you use E-cigarettes?", ECIG_OPTIONS, key="ECigaretteUsage", default=loaded.get("ECigaretteUsage"))
            SmokelessTobaccoUse = optional_select("How often do you use smokeless tobacco?", SMOKELESS_OPTIONS, key="SmokelessTobaccoUse", default=loaded.get("SmokelessTobaccoUse"))
            AlcoholDays = optional_number("Alcohol days code. 888 = no drinks, 101-107 = days/week, 201-230 = days/month", 0, 999, key="AlcoholDays", default=loaded.get("AlcoholDays", ""))
            PhysicalActivities = optional_select("In the past month, have you done any physical activity outside your job?", YES_NO_CODE_OPTIONS, key="PhysicalActivities", default=loaded.get("PhysicalActivities"))

        with col2:
            HadDiabetes = optional_text_select("Have you been diagnosed with Diabetes?", DIABETES_OPTIONS, key="HadDiabetes", default=loaded.get("HadDiabetes"))
            HadKidneyDisease = optional_text_select("Have you been diagnosed with Kidney Disease?", YES_NO_TEXT_OPTIONS, key="HadKidneyDisease", default=loaded.get("HadKidneyDisease"))
            HadStroke = optional_text_select("Have you had a Stroke?", YES_NO_TEXT_OPTIONS, key="HadStroke", default=loaded.get("HadStroke"))
            HadCOPD = optional_text_select("Have you been diagnosed with COPD?", YES_NO_TEXT_OPTIONS, key="HadCOPD", default=loaded.get("HadCOPD"))
            HadDepressiveDisorder = optional_text_select("Have you been diagnosed with Depressive Disorder?", YES_NO_TEXT_OPTIONS, key="HadDepressiveDisorder", default=loaded.get("HadDepressiveDisorder"))
            HadArthritis = optional_text_select("Have you been diagnosed with Arthritis?", YES_NO_TEXT_OPTIONS, key="HadArthritis", default=loaded.get("HadArthritis"))

    st.markdown("### Extra Context for AI Recommendations Only")
    st.caption("These fields are not used directly by the BRFSS ML prediction.")

    with st.expander("Wearable, Nutrition, and Symptoms", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            gender = st.text_input("Gender (optional)", value=loaded.get("gender", "") or "", placeholder="Leave blank if unknown", key="gender")
            job = st.text_input("Job (optional)", value=loaded.get("job", "") or "", placeholder="Example: student, office worker", key="job")
            height_cm = optional_number("Height cm (optional)", 80, 250, step=0.1, key="height_cm", default=loaded.get("height_cm", ""))
            weight_kg = optional_number("Weight kg (optional)", 20, 300, step=0.1, key="weight_kg", default=loaded.get("weight_kg", ""))
            avg_heart_rate = optional_number("Average heart rate, bpm", 30, 220, step=0.1, key="avg_heart_rate", default=loaded.get("avg_heart_rate", ""))
            resting_heart_rate = optional_number("Resting heart rate, bpm", 30, 180, step=0.1, key="resting_heart_rate", default=loaded.get("resting_heart_rate", ""))

        with col2:
            sleep_hours = optional_number("Average number of hours of sleep per night", 0, 24, step=0.1, key="sleep_hours", default=loaded.get("sleep_hours", ""))
            respiratory_rate = optional_number("Respiratory rate, breaths per minute", 5, 40, step=0.1, key="respiratory_rate", default=loaded.get("respiratory_rate", ""))
            calories_per_day = optional_number("Average calories consumed per day", 0, 10000, step=0.1, key="calories_per_day", default=loaded.get("calories_per_day", ""))
            protein_g = optional_number("Protein, grams per day", 0, 500, step=0.1, key="protein_g", default=loaded.get("protein_g", ""))
            carbs_g = optional_number("Carbs, grams per day", 0, 1000, step=0.1, key="carbs_g", default=loaded.get("carbs_g", ""))
            fat_g = optional_number("Fat, grams per day", 0, 500, step=0.1, key="fat_g", default=loaded.get("fat_g", ""))
            water_liters = optional_number("Water, liters per day", 0, 20, step=0.1, key="water_liters", default=loaded.get("water_liters", ""))

        symptoms = st.text_area(
            "Describe symptoms or concerns (optional)",
            value=loaded.get("symptoms", "") or "",
            placeholder="Example: chest tightness, fatigue, shortness of breath, poor sleep...",
            key="symptoms"
        )

    profile = {
        "username": username or None,
        "Sex": Sex,
        "AgeCategory": AgeCategory,
        "Education": Education,
        "Income": Income,
        "EmploymentStatus": EmploymentStatus,
        "MaritalStatus": MaritalStatus,
        "HomeOwnership": HomeOwnership,
        "GeneralHealth": GeneralHealth,
        "LastCheckup": LastCheckup,
        "Height": Height,
        "height_feet": height_feet,
        "height_inches": height_inches,
        "Weight": Weight,
        "Smoked100Cigarettes": Smoked100Cigarettes,
        "SmokerStatus": SmokerStatus,
        "ECigaretteUsage": ECigaretteUsage,
        "SmokelessTobaccoUse": SmokelessTobaccoUse,
        "AlcoholDays": AlcoholDays,
        "PhysicalActivities": PhysicalActivities,
        "HadDiabetes": HadDiabetes,
        "HadKidneyDisease": HadKidneyDisease,
        "HadStroke": HadStroke,
        "HadCOPD": HadCOPD,
        "HadDepressiveDisorder": HadDepressiveDisorder,
        "HadArthritis": HadArthritis,
        "gender": gender or None,
        "job": job or None,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
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

    if save_clicked:
        if not username:
            st.warning("Please enter a username before saving.")
        else:
            profiles[username] = profile
            save_profiles(profiles)
            st.success(f"Saved profile for {username}")

    important_fields = {
        "Sex": Sex,
        "AgeCategory": AgeCategory,
        "GeneralHealth": GeneralHealth,
        "Height": Height,
        "Weight": Weight,
        "PhysicalActivities": PhysicalActivities,
    }
    missing_important = [name for name, value in important_fields.items() if value is None]

    if st.button("Predict Risk"):
        if len(missing_important) >= 4:
            st.warning(
                "Many important BRFSS fields are missing, so this prediction may be less reliable."
            )

        st.session_state["profile"] = profile

        # Do not send local-only height_feet/height_inches to FastAPI schema.
        api_profile = {k: v for k, v in profile.items() if k not in ["height_feet", "height_inches"]}

        res = requests.post(f"{API_URL}/predict", json=api_profile)

        if res.status_code == 200:
            data = res.json()
            st.session_state["prediction"] = data["prediction"]
            st.session_state["predicted"] = True
            st.session_state["diet_plan_text"] = None
            st.session_state["risk_report"] = None
            st.session_state["lifestyle"] = None
            st.session_state["doctor_note"] = None
        else:
            st.error("Prediction failed.")
            st.write(res.text)

    if st.session_state["predicted"]:
        st.markdown("---")
        st.info("This is a BRFSS-only machine learning estimate/prototype, not a medical diagnosis.")
        if st.session_state["prediction"] == 1:
            st.error("Model Prediction: Higher Heart Disease Risk Estimate")
        else:
            st.success("Model Prediction: Lower Heart Disease Risk Estimate")


saved_profile = st.session_state.get("profile")
api_saved_profile = None
if saved_profile:
    api_saved_profile = {k: v for k, v in saved_profile.items() if k not in ["height_feet", "height_inches"]}


with diet_tab:
    if st.session_state["predicted"] and api_saved_profile:
        if st.button("Generate Diet Plan"):
            res = requests.post(f"{API_URL}/diet-plan", json=api_saved_profile)
            if res.status_code == 200:
                st.session_state["diet_plan_text"] = res.json()["diet_plan"]
            else:
                st.error("Diet plan generation failed.")
                st.write(res.text)

        if st.session_state["diet_plan_text"]:
            st.markdown("### Diet Plan")
            st.markdown(st.session_state["diet_plan_text"])
    else:
        st.info("Please complete your profile and run prediction first.")


with report_tab:
    if st.session_state["predicted"] and api_saved_profile:
        if st.button("Generate Risk Report"):
            res = requests.post(
                f"{API_URL}/risk-report",
                params={"prediction": st.session_state["prediction"], "language": language},
                json=api_saved_profile,
            )
            if res.status_code == 200:
                st.session_state["risk_report"] = res.json()["risk_report"]
            else:
                st.error("Risk report generation failed.")
                st.write(res.text)

        if st.session_state.get("risk_report"):
            st.markdown("### Risk Report")
            st.markdown(st.session_state["risk_report"])
    else:
        st.info("Please complete your profile and run prediction first.")


with lifestyle_tab:
    if st.session_state["predicted"] and api_saved_profile:
        if st.button("Lifestyle Suggestions"):
            res = requests.post(
                f"{API_URL}/lifestyle",
                params={"language": language},
                json=api_saved_profile,
            )
            if res.status_code == 200:
                st.session_state["lifestyle"] = res.json()["lifestyle"]
            else:
                st.error("Lifestyle generation failed.")
                st.write(res.text)

        if st.session_state.get("lifestyle"):
            st.markdown("### Lifestyle Advice")
            st.markdown(st.session_state["lifestyle"])
    else:
        st.info("Please complete your profile and run prediction first.")


with doctor_tab:
    if st.session_state["predicted"] and api_saved_profile:
        if st.button("Generate Doctor's Note"):
            res = requests.post(
                f"{API_URL}/doctor-note",
                params={"prediction": st.session_state["prediction"], "language": language},
                json=api_saved_profile,
            )
            if res.status_code == 200:
                st.session_state["doctor_note"] = res.json()["doctor_note"]
            else:
                st.error("Doctor note generation failed.")
                st.write(res.text)

        if st.session_state.get("doctor_note"):
            st.markdown("### Doctor's Note")
            st.markdown(st.session_state["doctor_note"])
    else:
        st.info("Please complete your profile and run prediction first.")


with st.sidebar:
    st.header("Diet & Medical Chatbot")
    user_input = st.chat_input("Ask anything")

    if user_input:
        res = requests.post(
            f"{API_URL}/chat",
            json={"message": user_input, "language": language},
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