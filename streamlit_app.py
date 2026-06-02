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

import streamlit as st
st.set_page_config(page_title="🫀 Heart Risk AI", layout="wide")  # must be first

import requests
import threading
import time
import uvicorn
import os
from src.mlproject.risk_adjustment import BRFSSRiskAdjuster
from src.mlproject.profile_manager  import ProfileManager

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = "http://127.0.0.1:8000"

# ── Auto-start FastAPI backend in a background thread ─────────────────────────
def start_backend():
    """Launch the FastAPI app on port 8000 in a daemon thread."""
    config = uvicorn.Config("app:app", host="127.0.0.1", port=8000, log_level="error")
    server = uvicorn.Server(config)
    server.run()

@st.cache_resource
def launch_backend():
    """
    Start the backend once per Streamlit session.
    cache_resource ensures this only runs once even across reruns.
    """
    thread = threading.Thread(target=start_backend, daemon=True)
    thread.start()
    # Give the server a moment to boot before Streamlit starts making requests
    time.sleep(2)
    return thread

launch_backend()

# ── Load shared resources once (cached across reruns) ─────────────────────────
@st.cache_resource
def load_adjuster():
    adj = BRFSSRiskAdjuster(os.path.join(os.path.dirname(os.path.abspath(__file__)), "brfss_survey_data_processed.csv"))
    adj.fit()
    return adj

@st.cache_resource
def load_pm():
    return ProfileManager(profiles_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles"))

adjuster = load_adjuster()
pm       = load_pm()

# ── Session state defaults ────────────────────────────────────────────────────
_defaults = {
    "predicted":       False,
    "prediction":      None,
    "base_prob":       None,
    "adjusted_result": None,
    "diet_plan_text":  None,
    "risk_report":     None,
    "lifestyle_advice":None,
    "doctor_note":     None,
    "chat_history":    [],
    "loaded_profile":  None,
    "username":        "",
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")
language = st.sidebar.selectbox(
    "🌐 Output language",
    ["English", "Hindi", "Spanish", "Tamil", "Bengali"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("👤 User Profile")

username_input = st.sidebar.text_input(
    "Username",
    value=st.session_state["username"],
    placeholder="e.g. john_doe"
).strip().lower()

# Detect username change → clear loaded profile
if username_input != st.session_state["username"]:
    st.session_state["username"]       = username_input
    st.session_state["loaded_profile"] = None

username = st.session_state["username"]

if username:
    if pm.profile_exists(username):
        st.sidebar.success(f"✅ Profile found for **{username}**")
        if st.sidebar.button("📂 Load saved profile"):
            st.session_state["loaded_profile"] = pm.load_profile(username)
            st.rerun()
    else:
        st.sidebar.info("🆕 New user — profile will be created on first prediction.")

# Convenience shortcuts for pre-filling
lp   = st.session_state["loaded_profile"] or {}
lp_c = lp.get("clinical",  {})
lp_l = lp.get("lifestyle", {})

if lp:
    st.sidebar.success("✅ Profile loaded — fields pre-filled.")

# ── Page title ────────────────────────────────────────────────────────────────
st.title("🫀 Personalised Heart Disease Risk Predictor")
st.caption("Fill in your clinical details and lifestyle information, then click **Predict**.")

# ── Helper widgets ────────────────────────────────────────────────────────────
def num_input(label, min_val=None, max_val=None, step=1.0, saved=None):
    """Text input that accepts numbers; returns int/float or None."""
    default_str = str(saved) if saved is not None else ""
    raw = st.text_input(label, value=default_str, placeholder="Leave blank if unknown")
    if raw.strip() == "":
        return None
    try:
        n = float(raw)
        if min_val is not None and n < min_val:
            st.warning(f"{label}: minimum is {min_val}"); return None
        if max_val is not None and n > max_val:
            st.warning(f"{label}: maximum is {max_val}"); return None
        return int(n) if step == 1 else n
    except ValueError:
        st.warning(f"{label}: please enter a valid number"); return None

def sel_input(label, options, saved=None):
    """Selectbox with an 'I don't know' guard. Returns selected value or None."""
    all_opts = ["I don't know"] + list(options)
    idx = 0
    if saved is not None and saved in options:
        idx = all_opts.index(saved)
    val = st.selectbox(label, all_opts, index=idx)
    return None if val == "I don't know" else val


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_clinical, tab_lifestyle, tab_diet, tab_report, tab_doctor, tab_history = st.tabs([
    "📋 Clinical Profile",
    "🏃 Lifestyle & BRFSS",
    "🥗 Diet Plan",
    "📊 Risk Report",
    "📄 Doctor's Note",
    "🕓 History",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Clinical Profile
# ══════════════════════════════════════════════════════════════════════════════
with tab_clinical:
    st.subheader("📋 Clinical Health Profile")
    st.caption("Leave any field blank if unknown.")

    with st.expander("🏠 Basic Information", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            age       = num_input("🎂 Age",        0,   120, saved=lp_c.get("age"))
            sex       = sel_input("♂️ Biological Sex", ["Male","Female"], saved=lp_c.get("sex"))
            height_cm = num_input("📏 Height (cm)", 80,  250, step=0.1, saved=lp_c.get("height_cm"))
            weight_kg = num_input("⚖️ Weight (kg)", 20,  300, step=0.1, saved=lp_c.get("weight_kg"))
        with c2:
            job    = st.text_input("💼 Occupation (optional)",     value=lp_c.get("job")    or "")
            gender = st.text_input("🪪 Gender identity (optional)", value=lp_c.get("gender") or "")

    with st.expander("💓 Heart & Medical Information", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            trestbps = num_input("🩺 Resting Blood Pressure (mmHg)", 60,  250, saved=lp_c.get("trestbps"))
            chol     = num_input("🧪 Cholesterol (mg/dL)",           50,  600, saved=lp_c.get("chol"))
            thalach  = num_input("❤️ Max Heart Rate Achieved",        40,  250, saved=lp_c.get("thalach"))
            oldpeak  = num_input("📉 ST Depression",                  0,   10,  step=0.1, saved=lp_c.get("oldpeak"))
        with c2:
            cp      = sel_input("💓 Chest Pain Type",
                        ["Typical Angina","Atypical Angina","Non-anginal","Asymptomatic"],
                        saved=lp_c.get("cp"))
            exang   = sel_input("🏃 Chest pain during exercise?",  ["No","Yes"],  saved=lp_c.get("exang"))
            fbs     = sel_input("🍬 Fasting blood sugar >120 mg/dL?", ["No","Yes"], saved=lp_c.get("fbs"))
            restecg = sel_input("📈 ECG Results",
                        ["Normal","ST-T Abnormality","Left Ventricular Hypertrophy"],
                        saved=lp_c.get("restecg"))
            slope   = sel_input("📊 Slope of ST Segment",
                        ["Upsloping","Flat","Downsloping"], saved=lp_c.get("slope"))
            ca      = sel_input("🦠 Major Vessels Coloured (0–3)", [0,1,2,3], saved=lp_c.get("ca"))
            thal    = sel_input("🦬 Thalassemia",
                        ["Normal","Fixed Defect","Reversible Defect"], saved=lp_c.get("thal"))

    with st.expander("⌚ Wearable Data (optional)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            avg_heart_rate     = num_input("Average heart rate (bpm)",  30, 220, step=0.1, saved=lp_c.get("avg_heart_rate"))
            resting_heart_rate = num_input("Resting heart rate (bpm)",  30, 180, step=0.1, saved=lp_c.get("resting_heart_rate"))
        with c2:
            sleep_hours      = num_input("Sleep (hours/night)",          0,  24, step=0.1, saved=lp_c.get("sleep_hours"))
            respiratory_rate = num_input("Respiratory rate (breaths/min)", 5, 40, step=0.1, saved=lp_c.get("respiratory_rate"))

    with st.expander("🥗 Nutrition (optional)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            calories_per_day = num_input("Calories/day",  0, 10000, saved=lp_c.get("calories_per_day"))
            protein_g        = num_input("Protein (g/day)", 0, 500, step=0.1, saved=lp_c.get("protein_g"))
            carbs_g          = num_input("Carbs (g/day)",   0, 1000, step=0.1, saved=lp_c.get("carbs_g"))
        with c2:
            fat_g        = num_input("Fat (g/day)",       0, 500, step=0.1, saved=lp_c.get("fat_g"))
            water_liters = num_input("Water (litres/day)", 0,  20, step=0.1, saved=lp_c.get("water_liters"))

    with st.expander("🗣️ Symptoms (optional)", expanded=False):
        symptoms = st.text_area(
            "Describe any symptoms or concerns",
            value=lp_c.get("symptoms") or "",
            placeholder="e.g. chest tightness, fatigue, shortness of breath..."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Lifestyle & BRFSS
# ══════════════════════════════════════════════════════════════════════════════
with tab_lifestyle:
    st.subheader("🏃 Lifestyle & Health Background")
    st.caption("Based on CDC BRFSS — these personalise your risk score.")

    with st.expander("🚬 Tobacco Use", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            smoked_100    = sel_input("Smoked 100+ cigarettes in your lifetime?",
                              ["Yes","No"], saved=lp_l.get("smoked_100_cigarettes_label"))
            smoker_status = sel_input("Do you currently smoke?",
                              ["Every day","Some days","Not at all"],
                              saved=lp_l.get("smoker_status_label"))
        with c2:
            ecig_usage = sel_input("Do you currently use e-cigarettes?",
                           ["Every day","Some days","Not at all","Never used e-cigarettes"],
                           saved=lp_l.get("ecig_usage_label"))
            smokeless  = sel_input("Do you use smokeless tobacco?",
                           ["Every day","Some days","Not at all"],
                           saved=lp_l.get("smokeless_label"))

    with st.expander("🍺 Alcohol Use", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            alcohol_drinker = sel_input("Do you drink alcoholic beverages?",
                                ["Yes","No"], saved=lp_l.get("alcohol_drinker_label"))
        with c2:
            alcohol_days_per_week = num_input("Alcohol drinking days per week (0–7)", 0, 7,
                                              saved=lp_l.get("alcohol_days_per_week_display"))

    with st.expander("🏋️ Physical Activity", expanded=True):
        physical_activities = sel_input(
            "In the past 30 days, did you do any physical activity outside your regular job?",
            ["Yes","No"], saved=lp_l.get("physical_activities_label")
        )

    with st.expander("🏥 Health Background", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            general_health = sel_input("How would you rate your general health?",
                               ["Excellent","Very Good","Good","Fair","Poor"],
                               saved=lp_l.get("general_health_label"))
            had_diabetes   = sel_input("Have you been told you have diabetes?",
                               ["Yes","Yes (pregnancy only)","No","Pre-diabetes / borderline"],
                               saved=lp_l.get("had_diabetes_label"))
            had_stroke     = sel_input("Have you ever had a stroke?",
                               ["Yes","No"], saved=lp_l.get("had_stroke_label"))
            had_copd       = sel_input("Have you been told you have COPD or emphysema?",
                               ["Yes","No"], saved=lp_l.get("had_copd_label"))
        with c2:
            had_angina     = sel_input("Have you been told you have angina or coronary heart disease?",
                               ["Yes","No"], saved=lp_l.get("had_angina_label"))
            had_kidney     = sel_input("Have you been told you have kidney disease?",
                               ["Yes","No"], saved=lp_l.get("had_kidney_label"))
            had_depression = sel_input("Have you been told you have a depressive disorder?",
                               ["Yes","No"], saved=lp_l.get("had_depression_label"))
            had_arthritis  = sel_input("Have you been told you have arthritis?",
                               ["Yes","No"], saved=lp_l.get("had_arthritis_label"))

    with st.expander("👤 Demographics", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            education = sel_input("Highest level of education",
                          ["Never attended school",
                           "Elementary (grades 1–8)",
                           "Some high school (grades 9–11)",
                           "High school graduate / GED",
                           "Some college or technical school",
                           "College graduate"],
                          saved=lp_l.get("education_label"))
            income    = sel_input("Annual household income",
                          ["Less than $10,000","$10,000–$15,000","$15,000–$20,000",
                           "$20,000–$25,000","$25,000–$35,000","$35,000–$50,000",
                           "$50,000–$75,000","$75,000–$100,000","$100,000–$150,000",
                           "$150,000–$200,000","$200,000 or more"],
                          saved=lp_l.get("income_label"))
        with c2:
            employment = sel_input("Employment status",
                           ["Employed for wages","Self-employed","Out of work (1+ year)",
                            "Out of work (<1 year)","Homemaker","Student",
                            "Retired","Unable to work"],
                           saved=lp_l.get("employment_label"))
            home_ownership = sel_input("Home ownership",
                               ["Own","Rent","Other arrangement"],
                               saved=lp_l.get("home_ownership_label"))
            marital_status = sel_input("Marital status",
                               ["Married","Divorced","Widowed","Separated",
                                "Never married","Member of unmarried couple"],
                               saved=lp_l.get("marital_status_label"))

    # ── Encode to BRFSS numeric codes ─────────────────────────────────────────
    SMOKER_MAP   = {"Every day":1,"Some days":2,"Not at all":3}
    ECIG_MAP     = {"Every day":1,"Some days":2,"Not at all":3,"Never used e-cigarettes":4}
    HEALTH_MAP   = {"Excellent":1,"Very Good":2,"Good":3,"Fair":4,"Poor":5}
    DIABETES_MAP = {"Yes":1,"Yes (pregnancy only)":2,"No":3,"Pre-diabetes / borderline":4}
    YESNO_MAP    = {"Yes":1,"No":2}
    ACTIVITY_MAP = {"Yes":1,"No":2}
    EDU_MAP = {
        "Never attended school":1,"Elementary (grades 1–8)":2,
        "Some high school (grades 9–11)":3,"High school graduate / GED":4,
        "Some college or technical school":5,"College graduate":6,
    }
    INCOME_MAP = {
        "Less than $10,000":1,"$10,000–$15,000":2,"$15,000–$20,000":3,
        "$20,000–$25,000":4,"$25,000–$35,000":5,"$35,000–$50,000":6,
        "$50,000–$75,000":7,"$75,000–$100,000":8,"$100,000–$150,000":9,
        "$150,000–$200,000":10,"$200,000 or more":11,
    }
    EMPLOY_MAP = {
        "Employed for wages":1,"Self-employed":2,"Out of work (1+ year)":3,
        "Out of work (<1 year)":4,"Homemaker":5,"Student":6,"Retired":7,"Unable to work":8,
    }
    HOME_MAP    = {"Own":1,"Rent":2,"Other arrangement":3}
    MARITAL_MAP = {
        "Married":1,"Divorced":2,"Widowed":3,
        "Separated":4,"Never married":5,"Member of unmarried couple":6,
    }

    def age_to_brfss(a):
        if a is None: return None
        for lo, hi, cat in [(18,24,1),(25,29,2),(30,34,3),(35,39,4),(40,44,5),
                             (45,49,6),(50,54,7),(55,59,8),(60,64,9),(65,69,10),
                             (70,74,11),(75,79,12)]:
            if lo <= a <= hi: return cat
        return 13

    # AlcoholDays: days/week → BRFSS encoding (101 + n)
    alcohol_days_encoded = (100 + int(alcohol_days_per_week)) if alcohol_days_per_week is not None else None

    # Build the lifestyle dict — numeric codes + label strings for saving
    lifestyle_data = {
        # Tobacco (numeric)
        "smoked_100_cigarettes":       YESNO_MAP.get(smoked_100),
        "smoker_status":               SMOKER_MAP.get(smoker_status),
        "ecigarette_usage":            ECIG_MAP.get(ecig_usage),
        "smokeless_tobacco_use":       SMOKER_MAP.get(smokeless),
        # Tobacco (labels for pre-fill)
        "smoked_100_cigarettes_label": smoked_100,
        "smoker_status_label":         smoker_status,
        "ecig_usage_label":            ecig_usage,
        "smokeless_label":             smokeless,
        # Alcohol (numeric)
        "alcohol_drinkers":            YESNO_MAP.get(alcohol_drinker),
        "alcohol_days":                alcohol_days_encoded,
        # Alcohol (labels)
        "alcohol_drinker_label":       alcohol_drinker,
        "alcohol_days_per_week_display": alcohol_days_per_week,
        # Activity (numeric + label)
        "physical_activities":         ACTIVITY_MAP.get(physical_activities),
        "physical_activities_label":   physical_activities,
        # Health status (numeric + label)
        "general_health":              HEALTH_MAP.get(general_health),
        "general_health_label":        general_health,
        "good_or_better_health":       1 if general_health in ["Excellent","Very Good","Good"]
                                       else (2 if general_health in ["Fair","Poor"] else None),
        # Comorbidities (numeric + label)
        "had_angina":                  YESNO_MAP.get(had_angina),
        "had_angina_label":            had_angina,
        "had_diabetes":                DIABETES_MAP.get(had_diabetes),
        "had_diabetes_label":          had_diabetes,
        "had_stroke":                  YESNO_MAP.get(had_stroke),
        "had_stroke_label":            had_stroke,
        "had_copd":                    YESNO_MAP.get(had_copd),
        "had_copd_label":              had_copd,
        "had_kidney_disease":          YESNO_MAP.get(had_kidney),
        "had_kidney_label":            had_kidney,
        "had_depressive_disorder":     YESNO_MAP.get(had_depression),
        "had_depression_label":        had_depression,
        "had_arthritis":               YESNO_MAP.get(had_arthritis),
        "had_arthritis_label":         had_arthritis,
        # Demographics (numeric + label)
        "sex":                         1 if sex == "Male" else (2 if sex == "Female" else None),
        "age_category":                age_to_brfss(age),
        "education":                   EDU_MAP.get(education),
        "education_label":             education,
        "income":                      INCOME_MAP.get(income),
        "income_label":                income,
        "employment_status":           EMPLOY_MAP.get(employment),
        "employment_label":            employment,
        "home_ownership":              HOME_MAP.get(home_ownership),
        "home_ownership_label":        home_ownership,
        "marital_status":              MARITAL_MAP.get(marital_status),
        "marital_status_label":        marital_status,
        # Height / weight for BMI (used by risk adjuster)
        "height":                      height_cm,
        "weight":                      weight_kg,
    }


# ── Clinical payload for API ───────────────────────────────────────────────────
clinical_data = {
    "age": age, "sex": sex, "gender": gender or None, "job": job or None,
    "height_cm": height_cm, "weight_kg": weight_kg,
    "cp": cp, "trestbps": trestbps, "chol": chol, "fbs": fbs,
    "restecg": restecg, "thalach": thalach, "exang": exang,
    "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
    "avg_heart_rate": avg_heart_rate, "resting_heart_rate": resting_heart_rate,
    "sleep_hours": sleep_hours, "respiratory_rate": respiratory_rate,
    "calories_per_day": calories_per_day, "protein_g": protein_g,
    "carbs_g": carbs_g, "fat_g": fat_g, "water_liters": water_liters,
    "symptoms": symptoms or None,
}

required_fields = {
    "age": age, "blood pressure": trestbps, "cholesterol": chol,
    "max heart rate": thalach, "ST depression": oldpeak,
    "chest pain type": cp, "ECG result": restecg,
    "ST slope": slope, "major vessels": ca, "thalassemia": thal,
}
missing = [name for name, val in required_fields.items() if val is None]


# ══════════════════════════════════════════════════════════════════════════════
# PREDICT BUTTON
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")

if st.button("🚑 Predict My Heart Disease Risk", type="primary", use_container_width=True):

    if len(missing) >= 5:
        st.warning(f"⚠️ {len(missing)} important clinical fields are missing — prediction may be less accurate.")

    # Stage 1: UCI clinical prediction via FastAPI
    with st.spinner("Running clinical prediction..."):
        res = requests.post(f"{API_URL}/predict", json=clinical_data)

    if res.status_code != 200:
        st.error("❌ Clinical prediction failed.")
        st.code(res.text)
        st.stop()

    data      = res.json()
    base_pred = data["prediction"]
    # Use probability if the endpoint returns it, otherwise fall back
    base_prob = data.get("probability", 0.72 if base_pred == 1 else 0.28)

    # Stage 2: BRFSS lifestyle adjustment
    with st.spinner("Applying lifestyle personalisation..."):
        adj_result = adjuster.adjust(base_prob, lifestyle_data)

    # Store results
    st.session_state["prediction"]      = base_pred
    st.session_state["base_prob"]       = base_prob
    st.session_state["adjusted_result"] = adj_result
    st.session_state["predicted"]       = True
    st.session_state["missing"]         = missing

    # Reset any previously generated outputs
    for k in ["diet_plan_text","risk_report","lifestyle_advice","doctor_note"]:
        st.session_state[k] = None

    # ── Save / update profile ──────────────────────────────────────────────
    if username:
        if pm.profile_exists(username):
            pm.update_clinical(username, clinical_data)
            pm.update_lifestyle(username, lifestyle_data)
        else:
            pm.save_profile(username, clinical_data, lifestyle_data)

        pm.add_prediction(
            username,
            prediction    = base_pred,
            probability   = base_prob,
            adjusted_risk = adj_result["adjusted_probability"],
            risk_level    = adj_result["risk_level"],
            factors       = adj_result["contributing_factors"],
        )
        st.sidebar.success(f"💾 Profile saved for **{username}**")


# ── Results display ────────────────────────────────────────────────────────────
if st.session_state["predicted"]:
    adj  = st.session_state["adjusted_result"]
    base = st.session_state["base_prob"]

    st.markdown("---")
    st.subheader("🔬 Your Results")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Clinical Risk (UCI model)", f"{base:.0%}")
    with c2:
        delta = (adj["adjusted_probability"] - base) * 100
        st.metric("Lifestyle-Adjusted Risk", f"{adj['adjusted_probability']:.0%}",
                  delta=f"{delta:+.1f}%")
    with c3:
        level = adj["risk_level"]
        if level == "High":
            st.error(f"⚠️ Risk Level: **{level}**")
        elif level == "Moderate":
            st.warning(f"🟡 Risk Level: **{level}**")
        else:
            st.success(f"✅ Risk Level: **{level}**")

    with st.expander("📋 Contributing lifestyle factors", expanded=True):
        for f in adj["contributing_factors"]:
            st.markdown(f"- {f}")

    st.caption("⚠️ This tool is not a medical diagnosis. Please consult a healthcare professional.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Diet Plan
# ══════════════════════════════════════════════════════════════════════════════
with tab_diet:
    if not st.session_state["predicted"]:
        st.info("⚠️ Run a prediction first.")
    else:
        if st.button("🥗 Generate Personalised Diet Plan"):
            with st.spinner("Generating diet plan..."):
                res = requests.post(f"{API_URL}/diet-plan", json=clinical_data)
            if res.status_code == 200:
                st.session_state["diet_plan_text"] = res.json()["diet_plan"]
            else:
                st.error("❌ Diet plan generation failed.")
        if st.session_state["diet_plan_text"]:
            st.markdown("### 🥗 Your Diet Plan")
            st.markdown(st.session_state["diet_plan_text"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Risk Report
# ══════════════════════════════════════════════════════════════════════════════
with tab_report:
    if not st.session_state["predicted"]:
        st.info("⚠️ Run a prediction first.")
    else:
        if st.button("📊 Generate Risk Report"):
            with st.spinner("Generating risk report..."):
                res = requests.post(
                    f"{API_URL}/risk-report",
                    params={"prediction": st.session_state["prediction"], "language": language},
                    json=clinical_data
                )
            if res.status_code == 200:
                st.session_state["risk_report"] = res.json()["risk_report"]
            else:
                st.error("❌ Risk report generation failed.")
        if st.session_state.get("risk_report"):
            st.markdown("### 📊 Risk Report")
            st.markdown(st.session_state["risk_report"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Doctor's Note
# ══════════════════════════════════════════════════════════════════════════════
with tab_doctor:
    if not st.session_state["predicted"]:
        st.info("⚠️ Run a prediction first.")
    else:
        if st.button("📄 Generate Doctor's Note"):
            with st.spinner("Generating doctor's note..."):
                res = requests.post(
                    f"{API_URL}/doctor-note",
                    params={"prediction": st.session_state["prediction"], "language": language},
                    json=clinical_data
                )
            if res.status_code == 200:
                st.session_state["doctor_note"] = res.json()["doctor_note"]
            else:
                st.error("❌ Doctor's note generation failed.")
        if st.session_state.get("doctor_note"):
            st.markdown("### 📄 Doctor's Note")
            st.markdown(st.session_state["doctor_note"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Prediction History
# ══════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.subheader("🕓 Prediction History")
    if not username:
        st.info("Enter a username in the sidebar to view your history.")
    elif not pm.profile_exists(username):
        st.info("No saved profile found for this username yet.")
    else:
        history = pm.get_prediction_history(username)
        if not history:
            st.info("No predictions logged yet for this user.")
        else:
            for i, record in enumerate(reversed(history)):
                label = f"{'⚠️ High' if record['risk_level']=='High' else ('🟡 Moderate' if record['risk_level']=='Moderate' else '✅ Low')} Risk — {record['timestamp'][:10]}"
                with st.expander(label):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Clinical Probability", f"{record['probability']:.0%}")
                    c2.metric("Adjusted Risk",        f"{record['adjusted_risk']:.0%}")
                    c3.metric("Risk Level",           record.get("risk_level","—"))
                    if record.get("factors"):
                        st.markdown("**Contributing factors:**")
                        for f in record["factors"]:
                            st.markdown(f"- {f}")


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar Chatbot — profile-aware
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("---")
    st.header("💬 Heart Health Chatbot")

    user_input = st.chat_input("Ask anything about your heart health...")

    if user_input:
        # Build context from saved profile + latest prediction
        context_parts = []

        if username and pm.profile_exists(username):
            summary = pm.get_summary(username)
            if summary:
                context_parts.append(summary)

        if st.session_state.get("adjusted_result"):
            adj = st.session_state["adjusted_result"]
            context_parts.append(
                f"Latest risk: {adj['adjusted_probability']:.0%} ({adj['risk_level']}). "
                f"Top factors: {'; '.join(adj['contributing_factors'][:3])}."
            )

        context = "\n".join(context_parts)
        full_msg = f"{context}\n\nUser question: {user_input}" if context else user_input

        res = requests.post(
            f"{API_URL}/chat",
            json={"message": full_msg, "language": language}
        )

        if res.status_code == 200:
            reply = res.json()["reply"]
            st.session_state.chat_history.append({"role": "user",      "content": user_input})
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
        else:
            st.error("Chatbot request failed.")

    for msg in st.session_state.chat_history[::-1]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])