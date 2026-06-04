# 🫀 Heart Disease Prediction & AI Diet Assistant

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python"/>
  <img src="https://img.shields.io/badge/Streamlit-latest-red.svg" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/Machine%20Learning-MLflow-green.svg" alt="MLflow"/>
  <img src="https://img.shields.io/badge/GenAI-LangChain%20%2B%20Groq-orange.svg" alt="GenAI"/>
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"/>
</p>

<p align="center">
  <strong>A full-stack AI-powered healthcare application for heart disease prediction and personalized diet & lifestyle guidance.</strong>
</p>

<p align="center">
  <a href="https://heart-diseasefrontend-5a7a8554af50.herokuapp.com/">🌐 Live App</a> •
  <a href="#-features">✨ Features</a> •
  <a href="#-architecture">🏗️ Architecture</a> •
  <a href="#-tech-stack">🛠️ Tech Stack</a> •
  <a href="#-quick-start">🚀 Quick Start</a>
</p>

---
## Running millen-dev

### Prerequisites
Make sure you have the following installed:
- Python 3.9+
- pip

### Setup

1. **Clone the repository and switch to the branch**
```bash
   git clone https://github.com/msbisrat/111project.git
   cd 111project
   git checkout millen-dev
```

2. **Create and activate a virtual environment**
```bash
   python -m venv .venv
   source .venv/bin/activate        # Mac/Linux
   .venv\Scripts\activate           # Windows
```

3. **Install dependencies**
```bash
   pip install -r requirements.txt
```

4. **Add your Groq API key**

   Get a free key at [console.groq.com](https://console.groq.com)
   Create a `.env` file in the project root: Write GROQ_API_KEY=your_key_here

5. **Download the data files**

   Download the following files from [Shared Drive](https://drive.google.com/drive/folders/1Do-WD_58u01GXPyg8ScpCBZtyY61Kdw6?usp=sharing) and place them in the project root:
   - `brfss_survey_data_processed.csv`
   - `notebook/data/cleaned_data.csv`
     - Ensure this file is in /notebook/data

6. **Run the training pipeline** (generates `artifact/model.pkl` and `artifact/preprocessor.pkl`)
```bash
   python src/mlproject/pipelines/training_pipelines.py
```

7. **Launch the app**
```bash
   streamlit run streamlit_app.py
```
   The FastAPI backend starts automatically in the background.

### Notes
- User profiles are saved locally in `profiles/` and are not committed to the repo.
- Do not commit `.env` or any `.csv` files.
---
## 🎯 Overview

<p align="center">
  <img src="https://github.com/user-attachments/assets/b70c4e38-0ab6-4368-a248-348351cab3db" width="900" />
</p>

This project is a production-style **Streamlit web application** that predicts heart disease risk using machine learning models and enhances the experience with **AI-powered diet, lifestyle, and medical recommendations**.

It combines:

* Classical ML for medical risk prediction
* Experiment tracking with MLflow
* Generative AI for personalized healthcare guidance
* A clean, interactive Streamlit frontend

Built during **May–July**, with focus on real-world usability and deployment readiness.

---

## ✨ Key Features

### 🔍 Heart Disease Risk Prediction

* Predicts likelihood of heart disease using medical attributes:
  age, cholesterol, blood pressure, chest pain type, ECG results, and more.
* Trained and evaluated multiple models:

  * Random Forest
  * XGBoost
  * CatBoost
* MLflow used for:

  * Experiment tracking
  * Model comparison
  * Model registry

### 📊 Data & Insights

* End-to-end ML pipeline:

  * Data collection
  * Preprocessing
  * Exploratory Data Analysis (EDA)
  * Feature engineering
* Interactive visualizations:

  * Feature importance
  * Prediction insights
* Visual tools:

  * Matplotlib
  * Plotly

### 🧠 AI-Powered Diet & Lifestyle Assistant

* Integrated **Groq LLaMA 3 (70B)** via LangChain for:

  * Personalized diet plans
  * Heart-risk reports
  * Lifestyle improvement suggestions
  * Doctor’s note drafting
  * Interactive health chatbot
* Supports **multilingual responses** for accessibility.

### 🥗 Personalized Diet Plan Generator

* Customized meal plans:

  * Breakfast
  * Lunch
  * Dinner
* Heart-friendly food recommendations
* Foods to avoid based on risk profile
* One-click **PDF download** using FPDF.

### 💬 Chatbot Diet Assistant

* Sidebar chatbot for real-time interaction
* Context-aware and memory-enabled conversations
* Designed for nutrition and heart-health queries.

### 🔒 Secure Data Handling

* Patient data stored using **MySQL**
* Environment variables managed with **dotenv**
* Sensitive credentials never hard-coded.

---

## 🏗️ Architecture

```
User Input (Streamlit UI)
        ↓
Data Preprocessing & Validation
        ↓
ML Models (RF / XGBoost / CatBoost)
        ↓
Prediction Output + Risk Score
        ↓
AI Layer (LangChain + Groq LLaMA 3)
        ↓
Diet Plan | Lifestyle Advice | Chatbot | PDF Report
```

MLflow runs alongside the pipeline to track metrics, parameters, and models.

---

## 🛠️ Tech Stack

| Layer               | Technology                      |
| ------------------- | ------------------------------- |
| Language            | Python 3.9+                     |
| Frontend            | Streamlit                       |
| ML Models           | Scikit-learn, XGBoost, CatBoost |
| Experiment Tracking | MLflow                          |
| Visualization       | Matplotlib, Plotly              |
| GenAI               | LangChain + Groq LLaMA 3 (70B)  |
| Database            | MySQL                           |
| PDF Export          | FPDF                            |
| Config Management   | python-dotenv                   |
| Deployment          | Heroku                          |

---

## 🚀 Quick Start

### Prerequisites

* Python 3.9+
* MySQL
* Groq API key

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/heart-disease-ai.git
cd heart-disease-ai

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file:

```
GROQ_API_KEY=your_groq_api_key
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DB=heart_disease
```

### Run the App

```bash
streamlit run app.py
```

---

## 📸 Screenshots

### Heart Disease Predictor

![Predictor](https://github.com/user-attachments/assets/b70c4e38-0ab6-4368-a248-348351cab3db)

### Personalized Diet Plan

![Diet](https://github.com/user-attachments/assets/eef71d7a-f9a0-411e-8cf4-84407374ebe1)

### AI Diet Assistant Chatbot

![Chatbot](https://github.com/user-attachments/assets/3d2d62d0-7ee2-4004-85d0-748feac111f6)

---

## 📅 Development Timeline

* May: Data analysis, ML model training, MLflow integration
* June: Streamlit UI, prediction pipeline, visualizations
* July: GenAI integration, chatbot, PDF export, deployment

---

## 🔮 Future Enhancements

* Medical report explanation with citations
* Wearable device data integration
* Doctor dashboard for patient monitoring
* Mobile-friendly UI improvements
* Expanded multilingual support

---

## 👨‍💻 Author

Vivek Kumar Gupta
AI Engineering Student | Building real-world ML & GenAI products

GitHub: [https://github.com/vivek34561](https://github.com/vivek34561)
LinkedIn: [https://linkedin.com/in/vivek-gupta-0400452b6](https://linkedin.com/in/vivek-gupta-0400452b6)
Portfolio: [https://resume-sepia-seven.vercel.app/](https://resume-sepia-seven.vercel.app/)

---

## 📄 License

MIT License © 2025 Vivek Kumar Gupta

---

If you want, next I can:

* Add a polished badges-only header version
* Optimize this README for recruiters
* Convert this into a case-study style project description
* Create a short “Why this project matters” section for interviews
