import json
from pathlib import Path

import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000"
PROFILE_FILE = Path("saved_profiles.json")

st.set_page_config(page_title="BRFSS Heart Risk & Diet AI 🫀", layout="wide")


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
    value = st.text_input(
        label,
        value=str(default) if default != "" else "",
        placeholder="Leave blank if unknown",
        key=key,
    )

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


def alcohol_days_input(default=None):
    """Collect alcohol use in plain language, then convert back to the BRFSS AlcoholDays code."""
    if default == 888:
        default_pattern = "No drinks in the past 30 days"
        default_days = 0
    elif isinstance(default, (int, float)) and 101 <= int(default) <= 107:
        default_pattern = "Days per week"
        default_days = int(default) - 100
    elif isinstance(default, (int, float)) and 201 <= int(default) <= 230:
        default_pattern = "Days in past 30 days"
        default_days = int(default) - 200
    else:
        default_pattern = "I don't know"
        default_days = 0

    pattern_options = [
        "I don't know",
        "No drinks in the past 30 days",
        "Days per week",
        "Days in past 30 days",
    ]

    pattern = st.selectbox(
        "Alcohol use",
        pattern_options,
        index=pattern_options.index(default_pattern),
        key="AlcoholUsePattern",
        help="This is saved as the original BRFSS AlcoholDays code for the model.",
    )

    if pattern == "I don't know":
        return None

    if pattern == "No drinks in the past 30 days":
        return 888

    if pattern == "Days per week":
        days = st.slider(
            "On average, how many days per week do you drink alcohol?",
            min_value=1,
            max_value=7,
            value=min(max(default_days, 1), 7),
            key="AlcoholDaysPerWeek",
        )
        return 100 + days

    days = st.slider(
        "On how many of the past 30 days did you drink alcohol?",
        min_value=1,
        max_value=30,
        value=min(max(default_days, 1), 30),
        key="AlcoholDaysPastMonth",
    )
    return 200 + days


def condition_value(label, selected_conditions):
    return "Yes" if label in selected_conditions else "No"


def age_to_brfss_category(age):
    if 18 <= age <= 24:
        return 1.0
    if 25 <= age <= 29:
        return 2.0
    if 30 <= age <= 34:
        return 3.0
    if 35 <= age <= 39:
        return 4.0
    if 40 <= age <= 44:
        return 5.0
    if 45 <= age <= 49:
        return 6.0
    if 50 <= age <= 54:
        return 7.0
    if 55 <= age <= 59:
        return 8.0
    if 60 <= age <= 64:
        return 9.0
    if 65 <= age <= 69:
        return 10.0
    if 70 <= age <= 74:
        return 11.0
    if 75 <= age <= 79:
        return 12.0
    return 13.0


def brfss_category_to_age(category_code):
    category_midpoints = {
        1.0: 21,
        2.0: 27,
        3.0: 32,
        4.0: 37,
        5.0: 42,
        6.0: 47,
        7.0: 52,
        8.0: 57,
        9.0: 62,
        10.0: 67,
        11.0: 72,
        12.0: 77,
        13.0: 80,
    }

    if category_code is None:
        return 35

    try:
        return category_midpoints.get(float(category_code), 35)
    except (TypeError, ValueError):
        return 35


def saved_height_to_total_inches(loaded_profile):
    feet = loaded_profile.get("height_feet")
    inches = loaded_profile.get("height_inches")

    try:
        if feet is not None and inches is not None:
            return int(float(feet) * 12 + float(inches))
    except (TypeError, ValueError):
        pass

    height_code = loaded_profile.get("Height")
    try:
        if height_code is not None:
            feet = int(float(height_code)) // 100
            inches = int(float(height_code)) % 100
            return feet * 12 + inches
    except (TypeError, ValueError):
        pass

    return 66


def saved_weight_or_default(loaded_profile):
    try:
        weight = int(float(loaded_profile.get("Weight")))
        return min(max(weight, 50), 700)
    except (TypeError, ValueError):
        return 160


def brfss_height_code(feet, inches):
    if feet is None or inches is None:
        return None
    return int(feet * 100 + inches)


def saved_sex_label(default=None):
    if default == 1.0:
        return "Male"
    if default == 2.0:
        return "Female"
    return "I don't know / prefer not to say"


def set_sex_choice(choice):
    st.session_state["SexChoice"] = choice

def format_height(total_inches):
    try:
        total_inches = int(total_inches)
        feet = total_inches // 12
        inches = total_inches % 12
        return f"{feet}'{inches}"
    except (TypeError, ValueError):
        return "Unknown"


SEX_OPTIONS = {"Male": 1.0, "Female": 2.0}

SEX_INPUT_OPTIONS = [
    "Male",
    "Female",
    "I don't know / prefer not to say",
]

AGE_OPTIONS = {
    "18-24": 1.0,
    "25-29": 2.0,
    "30-34": 3.0,
    "35-39": 4.0,
    "40-44": 5.0,
    "45-49": 6.0,
    "50-54": 7.0,
    "55-59": 8.0,
    "60-64": 9.0,
    "65-69": 10.0,
    "70-74": 11.0,
    "75-79": 12.0,
    "80+": 13.0,
}

EDUCATION_OPTIONS = {
    "Never attended school / kindergarten only": 1.0,
    "Grades 1-8": 2.0,
    "Grades 9-11": 3.0,
    "High school graduate / GED": 4.0,
    "Some college or technical school": 5.0,
    "College graduate": 6.0,
    "Refused / unknown": 9.0,
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
    "Refused": 99.0,
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
    "Refused": 9.0,
}

MARITAL_OPTIONS = {
    "Married": 1.0,
    "Divorced": 2.0,
    "Widowed": 3.0,
    "Separated": 4.0,
    "Never married": 5.0,
    "Unmarried couple": 6.0,
    "Refused": 9.0,
}

HOME_OPTIONS = {
    "Own": 1.0,
    "Rent": 2.0,
    "Other arrangement": 3.0,
    "Don't know": 7.0,
    "Refused": 9.0,
}

GENERAL_HEALTH_OPTIONS = {
    "Excellent": 1.0,
    "Very good": 2.0,
    "Good": 3.0,
    "Fair": 4.0,
    "Poor": 5.0,
    "Don't know": 7.0,
    "Refused": 9.0,
}

YES_NO_CODE_OPTIONS = {
    "Yes": 1.0,
    "No": 2.0,
    "Don't know": 7.0,
    "Refused": 9.0,
}

LAST_CHECKUP_OPTIONS = {
    "Within past year": 1.0,
    "Within past 2 years": 2.0,
    "Within past 5 years": 3.0,
    "5 or more years ago": 4.0,
    "Don't know": 7.0,
    "Never": 8.0,
    "Refused": 9.0,
}

SMOKER_STATUS_OPTIONS = {
    "Current smoker - every day": 1.0,
    "Current smoker - some days": 2.0,
    "Former smoker": 3.0,
    "Never smoked": 4.0,
    "Don't know / refused": 9.0,
}

ECIG_OPTIONS = {
    "Use every day": 1.0,
    "Use some days": 2.0,
    "Not at all": 3.0,
    "Never used": 4.0,
    "Don't know": 7.0,
    "Refused": 9.0,
}

SMOKELESS_OPTIONS = {
    "Use every day": 1.0,
    "Use some days": 2.0,
    "Not at all": 3.0,
    "Don't know": 7.0,
    "Refused": 9.0,
}

YES_NO_TEXT_OPTIONS = ["Yes", "No"]

DIABETES_OPTIONS = [
    "Yes",
    "No",
    "No, pre-diabetes or borderline diabetes",
    "Yes, only during pregnancy",
]

DIAGNOSIS_LABEL_TO_KEY = {
    "Diabetes": "HadDiabetes",
    "Kidney disease": "HadKidneyDisease",
    "Stroke": "HadStroke",
    "COPD": "HadCOPD",
    "Arthritis": "HadArthritis",
    "Depressive disorder": "HadDepressiveDisorder",
}

DIAGNOSIS_OPTIONS = list(DIAGNOSIS_LABEL_TO_KEY.keys())


for key in [
    "predicted",
    "prediction",
    "diet_plan_text",
    "risk_report",
    "lifestyle",
    "doctor_note",
    "chat_history",
    "profile",
]:
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
    st.caption("Complete the fields you know. Optional questions can be skipped without blocking the risk estimate.")

    profiles = load_profiles()

    col_a, col_b, col_c = st.columns([3, 1, 1], gap="medium")
    with col_a:
        username = st.text_input(
            "Username for saving/loading profile",
            placeholder="Example: Sarah",
        )
    with col_b:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        load_clicked = st.button("Load Profile", use_container_width=True)
    with col_c:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        save_clicked = st.button("Save Profile", use_container_width=True)

    loaded = {}
    if load_clicked and username:
        loaded = profiles.get(username, {})

        if loaded:
            st.session_state["loaded_profile"] = loaded

            selectbox_option_maps = {
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

            for key, value in loaded.items():
                if value is None:
                    continue

                if key in selectbox_option_maps:
                    for label, code in selectbox_option_maps[key].items():
                        if code == value:
                            st.session_state[key] = label
                            break
                elif key in [
                    "height_cm",
                    "weight_kg",
                    "avg_heart_rate",
                    "resting_heart_rate",
                    "sleep_hours",
                    "respiratory_rate",
                    "calories_per_day",
                    "protein_g",
                    "carbs_g",
                    "fat_g",
                    "water_liters",
                    "gender",
                    "job",
                    "symptoms",
                ]:
                    st.session_state[key] = str(value)

            if loaded.get("Sex") is not None:
                st.session_state["SexChoice"] = saved_sex_label(loaded.get("Sex"))

            if loaded.get("AgeCategory") is not None:
                st.session_state["AgeSlider"] = brfss_category_to_age(loaded.get("AgeCategory"))

            st.session_state["HeightSlider"] = saved_height_to_total_inches(loaded)
            st.session_state["WeightSlider"] = saved_weight_or_default(loaded)

            diagnosed_conditions = []
            for display_label, profile_key in DIAGNOSIS_LABEL_TO_KEY.items():
                if loaded.get(profile_key) == "Yes":
                    diagnosed_conditions.append(display_label)
            st.session_state["DiagnosedConditions"] = diagnosed_conditions

            if isinstance(loaded.get("AlcoholDays"), (int, float)):
                alcohol_code = int(loaded["AlcoholDays"])
                if alcohol_code == 888:
                    st.session_state["AlcoholUsePattern"] = "No drinks in the past 30 days"
                elif 101 <= alcohol_code <= 107:
                    st.session_state["AlcoholUsePattern"] = "Days per week"
                    st.session_state["AlcoholDaysPerWeek"] = alcohol_code - 100
                elif 201 <= alcohol_code <= 230:
                    st.session_state["AlcoholUsePattern"] = "Days in past 30 days"
                    st.session_state["AlcoholDaysPastMonth"] = alcohol_code - 200

            st.success(f"Loaded profile for {username}")
            st.rerun()

        else:
            st.warning("No saved profile found for this username.")

    loaded = st.session_state.get("loaded_profile", {})

    st.markdown("### Prediction Profile")
    st.caption("These questions are grouped for readability, but the internal keys still match the BRFSS/FastAPI profile fields.")

    with st.expander("Demographics 👤", expanded=True):
        st.markdown("**Sex**")

        if "SexChoice" not in st.session_state:
            st.session_state["SexChoice"] = saved_sex_label(loaded.get("Sex"))

        sex_col1, sex_col2, sex_col3 = st.columns(3, gap="medium")

        with sex_col1:
            male_selected = st.session_state["SexChoice"] == "Male"
            if st.button(
                "Male",
                use_container_width=True,
                type="primary" if male_selected else "secondary",
                key="SexMaleButton",
            ):
                set_sex_choice("Male")
                st.rerun()

        with sex_col2:
            female_selected = st.session_state["SexChoice"] == "Female"
            if st.button(
                "Female",
                use_container_width=True,
                type="primary" if female_selected else "secondary",
                key="SexFemaleButton",
            ):
                set_sex_choice("Female")
                st.rerun()

        with sex_col3:
            unknown_selected = st.session_state["SexChoice"] == "I don't know / prefer not to say"
            if st.button(
                "Prefer not to say",
                use_container_width=True,
                type="primary" if unknown_selected else "secondary",
                key="SexUnknownButton",
            ):
                set_sex_choice("I don't know / prefer not to say")
                st.rerun()

        Sex = SEX_OPTIONS.get(st.session_state["SexChoice"])
        st.caption(f"Selected sex: {st.session_state['SexChoice']}")

        age_value = st.slider(
            "Age",
            min_value=18,
            max_value=100,
            value=brfss_category_to_age(loaded.get("AgeCategory")),
            step=1,
            key="AgeSlider",
            help="Slide right for older age.",
        )
        AgeCategory = age_to_brfss_category(age_value)
        st.caption(f"Selected age: {age_value}")

        height_options = list(range(36, 97))
        height_labels = [format_height(inches) for inches in height_options]

        default_height_inches = saved_height_to_total_inches(loaded)
        default_height_inches = min(max(default_height_inches, 36), 96)
        default_height_label = format_height(default_height_inches)

        height_label = st.select_slider(
            "Height",
            options=height_labels,
            value=default_height_label,
            key="HeightSlider",
            help="Slide right for taller height.",
        )

        height_inches_total = height_options[height_labels.index(height_label)]
        height_feet = height_inches_total // 12
        height_inches = height_inches_total % 12
        Height = brfss_height_code(height_feet, height_inches)

        st.caption(f"Selected height: {height_label}")

        Weight = st.slider(
            "Weight — pounds",
            min_value=50,
            max_value=700,
            value=saved_weight_or_default(loaded),
            step=1,
            key="WeightSlider",
            help="Slide right for higher weight.",
        )
        st.caption(f"Selected weight: {Weight} lb")

        st.markdown("---")
        st.caption("Optional demographic details")

        col1, col2 = st.columns(2, gap="large")

        with col1:
            Education = optional_select(
                "Education (optional)",
                EDUCATION_OPTIONS,
                key="Education",
                default=loaded.get("Education"),
            )

            MaritalStatus = optional_select(
                "Marital Status (optional)",
                MARITAL_OPTIONS,
                key="MaritalStatus",
                default=loaded.get("MaritalStatus"),
            )

            Income = optional_select(
                "Income (optional)",
                INCOME_OPTIONS,
                key="Income",
                default=loaded.get("Income"),
            )

        with col2:
            EmploymentStatus = optional_select(
                "Employment Status (optional)",
                EMPLOYMENT_OPTIONS,
                key="EmploymentStatus",
                default=loaded.get("EmploymentStatus"),
            )

            HomeOwnership = optional_select(
                "Home Ownership (optional)",
                HOME_OPTIONS,
                key="HomeOwnership",
                default=loaded.get("HomeOwnership"),
            )

    with st.expander("General Health 🩺", expanded=True):
        col1, col2 = st.columns(2, gap="large")

        with col1:
            GeneralHealth = optional_select(
                "General Health",
                GENERAL_HEALTH_OPTIONS,
                key="GeneralHealth",
                default=loaded.get("GeneralHealth"),
            )
            SmokerStatus = optional_select(
                "Smoker Status",
                SMOKER_STATUS_OPTIONS,
                key="SmokerStatus",
                default=loaded.get("SmokerStatus"),
            )
            ECigaretteUsage = optional_select(
                "E-cigarette Usage",
                ECIG_OPTIONS,
                key="ECigaretteUsage",
                default=loaded.get("ECigaretteUsage"),
            )
            AlcoholDays = alcohol_days_input(default=loaded.get("AlcoholDays"))

        with col2:
            LastCheckup = optional_select(
                "Last Routine Checkup",
                LAST_CHECKUP_OPTIONS,
                key="LastCheckup",
                default=loaded.get("LastCheckup"),
            )
            Smoked100Cigarettes = optional_text_select(
                "Smoked at least 100 cigarettes?",
                YES_NO_TEXT_OPTIONS,
                key="Smoked100Cigarettes",
                default=loaded.get("Smoked100Cigarettes"),
            )
            SmokelessTobaccoUse = optional_select(
                "Smokeless Tobacco Use",
                SMOKELESS_OPTIONS,
                key="SmokelessTobaccoUse",
                default=loaded.get("SmokelessTobaccoUse"),
            )
            PhysicalActivities = optional_select(
                "Physical Activity in Past Month",
                YES_NO_CODE_OPTIONS,
                key="PhysicalActivities",
                default=loaded.get("PhysicalActivities"),
            )

    with st.expander("Health Conditions ✅", expanded=True):
        default_conditions = [
            label
            for label, profile_key in DIAGNOSIS_LABEL_TO_KEY.items()
            if loaded.get(profile_key) == "Yes"
        ]

        selected_conditions = st.multiselect(
            "Select any conditions you have been diagnosed with",
            DIAGNOSIS_OPTIONS,
            default=default_conditions,
            key="DiagnosedConditions",
            help="Leave this blank if none apply or if you do not know.",
        )

        st.caption("Selected conditions are saved back into the original yes/no fields expected by the backend.")

        HadDiabetes = condition_value("Diabetes", selected_conditions)
        HadKidneyDisease = condition_value("Kidney disease", selected_conditions)
        HadStroke = condition_value("Stroke", selected_conditions)
        HadCOPD = condition_value("COPD", selected_conditions)
        HadArthritis = condition_value("Arthritis", selected_conditions)
        HadDepressiveDisorder = condition_value("Depressive disorder", selected_conditions)

    st.markdown("### Extra Context for AI Recommendations Only ✨")
    st.caption("These optional fields are not sent as core BRFSS model inputs, but they can make AI-generated recommendations more useful.")

    with st.expander("Lifestyle Details, Nutrition, and Symptoms 📝", expanded=False):
        col1, col2 = st.columns(2, gap="large")

        with col1:
            gender = st.text_input(
                "Gender (optional)",
                value=loaded.get("gender", "") or "",
                placeholder="Leave blank if unknown",
                key="gender",
            )
            job = st.text_input(
                "Job (optional)",
                value=loaded.get("job", "") or "",
                placeholder="Example: student, office worker",
                key="job",
            )
            height_cm = optional_number(
                "Height, cm (optional)",
                80,
                250,
                step=0.1,
                key="height_cm",
                default=loaded.get("height_cm", ""),
            )
            weight_kg = optional_number(
                "Weight, kg (optional)",
                20,
                300,
                step=0.1,
                key="weight_kg",
                default=loaded.get("weight_kg", ""),
            )
            avg_heart_rate = optional_number(
                "Average heart rate, bpm (optional)",
                30,
                220,
                step=0.1,
                key="avg_heart_rate",
                default=loaded.get("avg_heart_rate", ""),
            )
            resting_heart_rate = optional_number(
                "Resting heart rate, bpm (optional)",
                30,
                180,
                step=0.1,
                key="resting_heart_rate",
                default=loaded.get("resting_heart_rate", ""),
            )

        with col2:
            sleep_hours = optional_number(
                "Average sleep per night, hours (optional)",
                0,
                24,
                step=0.1,
                key="sleep_hours",
                default=loaded.get("sleep_hours", ""),
            )
            respiratory_rate = optional_number(
                "Respiratory rate, breaths/min (optional)",
                5,
                40,
                step=0.1,
                key="respiratory_rate",
                default=loaded.get("respiratory_rate", ""),
            )
            calories_per_day = optional_number(
                "Calories per day (optional)",
                0,
                10000,
                step=0.1,
                key="calories_per_day",
                default=loaded.get("calories_per_day", ""),
            )
            protein_g = optional_number(
                "Protein, grams/day (optional)",
                0,
                500,
                step=0.1,
                key="protein_g",
                default=loaded.get("protein_g", ""),
            )
            carbs_g = optional_number(
                "Carbs, grams/day (optional)",
                0,
                1000,
                step=0.1,
                key="carbs_g",
                default=loaded.get("carbs_g", ""),
            )
            fat_g = optional_number(
                "Fat, grams/day (optional)",
                0,
                500,
                step=0.1,
                key="fat_g",
                default=loaded.get("fat_g", ""),
            )
            water_liters = optional_number(
                "Water, liters/day (optional)",
                0,
                20,
                step=0.1,
                key="water_liters",
                default=loaded.get("water_liters", ""),
            )

        symptoms = st.text_area(
            "Symptoms or concerns (optional)",
            value=loaded.get("symptoms", "") or "",
            placeholder="Example: chest tightness, fatigue, shortness of breath, poor sleep...",
            key="symptoms",
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
        "GoodOrBetterHealth": None,
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

    if st.button("Predict Risk", type="primary"):
        if len(missing_important) >= 4:
            st.warning("Many important BRFSS fields are missing, so this prediction may be less reliable.")

        st.session_state["profile"] = profile

        api_profile = {
            k: v
            for k, v in profile.items()
            if k not in ["height_feet", "height_inches"]
        }

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
            st.error("Model Estimate: High Risk of Heart Disease")
        else:
            st.success("Model Estimate: Low Risk of Heart Disease")


saved_profile = st.session_state.get("profile")
api_saved_profile = None

if saved_profile:
    api_saved_profile = {
        k: v
        for k, v in saved_profile.items()
        if k not in ["height_feet", "height_inches"]
    }


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
                params={
                    "prediction": st.session_state["prediction"],
                    "language": language,
                },
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
                params={
                    "prediction": st.session_state["prediction"],
                    "language": language,
                },
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