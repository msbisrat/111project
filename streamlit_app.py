import streamlit as st
import requests

# ------------------------- Backend URL -------------------------
API_URL = "https://heart-disease-prediction-ai-diet-gdob.onrender.com/"   # Change if deployed

st.set_page_config(page_title="🪀 Heart Risk & Diet AI", layout="wide")

st.sidebar.header("🔑 Configuration")
language = st.sidebar.selectbox("🌐 Select Output Language", ["English", "Hindi", "Spanish", "Tamil", "Bengali"])

st.title("🫀 Risk Of Heart Disease Predictor & Diet Assistant")

# ------------------------- Session State -------------------------
for key in ["predicted", "prediction", "diet_plan_text", "risk_report", "lifestyle", "doctor_note", "chat_history"]:
    if key not in st.session_state:
        st.session_state[key] = False if key == "predicted" else [] if key == "chat_history" else None

# ------------------------- Tabs -------------------------
profile_tab, diet_tab, report_tab, lifestyle_tab, doctor_tab = st.tabs(
    ["📋 Profile", "🥗 Diet Plan", "🗾 Risk Report", "🏃 Lifestyle", "📄 Doctor's Note"]
)

# ------------------------- Profile Tab -------------------------
with profile_tab:
    st.subheader("📋 Your Health Profile")

    with st.expander("🏠 Lifestyle & Demographics", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("🎂 Age", 20, 90, 45)
            sex = st.radio("♂️ Biological Sex", ["Male", "Female"])
        with col2:
            exang = st.radio("🏃 Chest pain during exercise?", ["No", "Yes"])
            fbs = st.radio("🍬 Fasting blood sugar > 120 mg/dL?", ["No", "Yes"])

    with st.expander("💓 Vitals & Tests", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            trestbps = st.slider("🩺 Resting Blood Pressure (mm Hg)", 80, 200, 120)
            chol = st.slider("🧪 Cholesterol Level (mg/dL)", 100, 400, 220)
            thalach = st.slider("❤️ Max Heart Rate Achieved", 60, 210, 150)
        with col2:
            oldpeak = st.slider("📉 ST Depression (Exercise vs Rest)", 0.0, 6.0, 1.0, 0.1)
            restecg = st.selectbox("📈 ECG Results", ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"])
            slope = st.selectbox("📊 Slope of ST Segment", ["Upsloping", "Flat", "Downsloping"])

    with st.expander("🧬 Medical History", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            cp = st.selectbox("💓 Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"])
        with col2:
            ca = st.selectbox("🦠 Number of Major Vessels Colored", [0, 1, 2, 3])
            thal = st.selectbox("🦬 Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

    # Prepare request payload
    profile = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }

    if st.button("🚑 Predict Risk"):
        res = requests.post(f"{API_URL}/predict", json=profile)
        if res.status_code == 200:
            data = res.json()
            st.session_state["prediction"] = data["prediction"]
            st.session_state["predicted"] = True
        else:
            st.error("❌ Prediction failed.")

    if st.session_state["predicted"]:
        st.markdown("---")
        if st.session_state["prediction"] == 1:
            st.error("⚠️ **High Risk of Heart Disease Detected!** Consult a cardiologist.")
        else:
            st.success("✅ **Low Risk of Heart Disease. Keep maintaining your health!**")

# ------------------------- Diet Plan Tab -------------------------
with diet_tab:
    if st.session_state["predicted"]:
        if st.button("🥗 Generate Diet Plan"):
            res = requests.post(f"{API_URL}/diet-plan", json=profile)
            if res.status_code == 200:
                st.session_state["diet_plan_text"] = res.json()["diet_plan"]
            else:
                st.error("❌ Diet plan generation failed.")

        if st.session_state["diet_plan_text"]:
            st.markdown("### 🥗 Diet Plan")
            st.markdown(st.session_state["diet_plan_text"])
            

    else:
        st.info("⚠️ Please complete your profile and run prediction first.")

# ------------------------- Risk Report Tab -------------------------
with report_tab:
    if st.session_state["predicted"]:
        if st.button("🗾 Generate Risk Report"):
            res = requests.post(f"{API_URL}/risk-report", params={"prediction": st.session_state["prediction"], "language": language}, json=profile)
            if res.status_code == 200:
                st.session_state["risk_report"] = res.json()["risk_report"]

        if st.session_state.get("risk_report"):
            st.markdown("### 🗾 Risk Report")
            st.markdown(st.session_state["risk_report"])
            

    else:
        st.info("⚠️ Please complete your profile and run prediction first.")

# ------------------------- Lifestyle Tab -------------------------
with lifestyle_tab:
    if st.session_state["predicted"]:
        if st.button("🏃 Lifestyle Suggestions"):
            res = requests.post(f"{API_URL}/lifestyle", params={"language": language}, json=profile)
            if res.status_code == 200:
                st.session_state["lifestyle"] = res.json()["lifestyle"]

        if st.session_state.get("lifestyle"):
            st.markdown("### 🏃 Lifestyle Advice")
            st.markdown(st.session_state["lifestyle"])

    else:
        st.info("⚠️ Please complete your profile and run prediction first.")

# ------------------------- Doctor's Note Tab -------------------------
with doctor_tab:
    if st.session_state["predicted"]:
        if st.button("📄 Generate Doctor's Note"):
            res = requests.post(f"{API_URL}/doctor-note", params={"prediction": st.session_state["prediction"], "language": language}, json=profile)
            if res.status_code == 200:
                st.session_state["doctor_note"] = res.json()["doctor_note"]

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
        res = requests.post(f"{API_URL}/chat", json={"message": user_input, "language": language})
        if res.status_code == 200:
            reply = res.json()["reply"]
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.chat_history.append({"role": "assistant", "content": reply})

    for msg in st.session_state.chat_history[::-1]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 🪡 Chat History")
        for msg in reversed(st.session_state.chat_history):
            st.markdown(f"**{msg['role'].capitalize()}**: {msg['content']}")