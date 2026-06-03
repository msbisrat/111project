# Heart Disease Risk Predictor & Diet Assistant AI

A machine learning health application that estimates heart disease risk from BRFSS-style health, demographic, lifestyle, and chronic-condition survey inputs. The app also includes an AI assistant layer that can generate diet suggestions, risk explanations, lifestyle guidance, and a doctor's-note style summary.

> **Medical disclaimer:** This project is for educational use only. It does not diagnose disease, replace a clinician, or provide emergency medical care.

---

## Project Summary

This project focuses on predicting whether a user is likely to be in the heart disease risk class using BRFSS survey-derived features. The model was trained on `brfss_2024_eda_processed.csv`, with `HadHeartDisease` as the target variable.

The project combines:

- A BRFSS-trained binary classification model
- A preprocessing pipeline saved as a pickle artifact
- A Streamlit/FastAPI app interface
- A Groq-powered chatbot and recommendation layer
- Generated outputs for diet plans, risk reports, lifestyle advice, and doctor-note summaries

---

## Data

The model uses BRFSS 2024 processed survey data. BRFSS captures health behavior, chronic condition, demographic, and lifestyle information across the United States.

### Target Variable

The prediction target is:

- `HadHeartDisease`
  - `Yes` mapped to `1`
  - `No` mapped to `0`

The task is binary classification:

- `1` = had heart disease / higher heart disease risk class
- `0` = no heart disease / lower heart disease risk class

---

## BRFSS Features Used

The final BRFSS training script uses 24 input features.

### Demographics and socioeconomic variables

- `Sex`
- `AgeCategory`
- `Education`
- `Income`
- `EmploymentStatus`
- `MaritalStatus`
- `HomeOwnership`

### General health and body measures

- `GeneralHealth`
- `GoodOrBetterHealth`
- `LastCheckup`
- `Height`
- `Weight`

### Health behaviors

- `Smoked100Cigarettes`
- `SmokerStatus`
- `ECigaretteUsage`
- `SmokelessTobaccoUse`
- `AlcoholDays`
- `PhysicalActivities`

### Existing conditions and comorbidities

- `HadDiabetes`
- `HadKidneyDisease`
- `HadStroke`
- `HadCOPD`
- `HadDepressiveDisorder`
- `HadArthritis`

These features were chosen because the project EDA found that smoking status, physical activity, alcohol use, diabetes, stroke, other comorbidities, and socioeconomic variables were important indicators of heart disease risk.

---

## Machine Learning Pipeline

The current BRFSS model is trained using `train_brfss_option_b.py`.

### 1. Data loading

The script loads:

```text
brfss_2024_eda_processed.csv
```

It keeps only the selected BRFSS feature columns and the `HadHeartDisease` target column.

### 2. Target conversion

The target is converted from text labels to binary labels:

```text
Yes -> 1
No  -> 0
```

Rows with missing target values are removed.

### 3. Train/validation/test split

The dataset is split with stratification to preserve the heart disease class distribution:

- 60% training
- 20% validation
- 20% test

The validation set is used for model selection. The test set is only used for final evaluation.

### 4. Preprocessing

The preprocessing pipeline uses a `ColumnTransformer` with separate numeric and categorical pipelines.

Numeric pipeline:

```text
SimpleImputer(strategy="median")
StandardScaler()
```

Categorical pipeline:

```text
SimpleImputer(strategy="most_frequent")
OneHotEncoder(handle_unknown="ignore")
```

The trained preprocessor is saved as:

```text
artifacts/preprocessor.pkl
artifact/preprocessor.pkl
```

### 5. Model training

The project compares untuned baseline models against tuned models with class imbalance handling.

Baseline models:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

Tuned models:

- Logistic Regression with `class_weight="balanced"`
- Decision Tree with limited depth and `class_weight="balanced"`
- Random Forest with `class_weight="balanced"`
- Gradient Boosting with balanced sample weights

### 6. Model selection

The dataset is imbalanced, so the final model is not selected by accuracy alone. The selection priority is:

1. F1-score for the positive heart disease class
2. Recall for the positive heart disease class
3. Precision for the positive heart disease class
4. ROC-AUC

This matters because a high-accuracy model can still miss many positive heart disease cases.

---

## Final Selected Model

The final selected model is a tuned Random Forest classifier.

```text
Model: Random Forest
n_estimators: 100
max_depth: 16
min_samples_leaf: 20
class_weight: balanced
```

The final model is saved as:

```text
artifacts/model.pkl
```

---

## Evaluation Results

### Validation performance

```text
Accuracy:          0.7611
Precision class 1: 0.2458
Recall class 1:    0.7506
F1 class 1:        0.3703
ROC-AUC:           0.8356
```

### Final test performance

```text
Accuracy:          0.7636
Precision class 1: 0.2482
Recall class 1:    0.7527
F1 class 1:        0.3733
ROC-AUC:           0.8382
```

### Final test confusion matrix

```text
                  Predicted No   Predicted Yes
Actual No              62,722          19,303
Actual Yes              2,094           6,374
```

The model has strong recall for the heart disease class, meaning it catches many positive cases. The tradeoff is lower precision, which means it also produces more false positives.

---

## Architecture

The project presentation describes the system as a two-stage architecture: a clinical prediction stage and a BRFSS/lifestyle adjustment stage. The current codebase contains both the older UCI-style clinical pipeline pieces and the newer BRFSS training script.


```markdown
![Two-stage pipeline](docs/pipeline_architecture.png)
```

![Two-stage pipeline](docs/pipeline_architecture.png)

### Current code architecture

```text
User input
   ↓
Streamlit frontend
   ↓
FastAPI backend
   ↓
Preprocessing pipeline
   ↓
Random Forest BRFSS model
   ↓
Risk prediction
   ↓
Groq AI assistant layer
   ↓
Diet plan, risk report, lifestyle advice, doctor note, chatbot response
```

---

## App Features

The application includes:

- Heart disease risk prediction
- Diet plan generation
- Risk report generation
- Lifestyle advice
- Doctor's-note style summary
- Sidebar chatbot
- Multilingual response support

The AI assistant uses Groq chat completions to generate natural-language responses from the user profile and prediction result.

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- FastAPI
- Groq API
- FPDF
- python-dotenv
- Matplotlib
- MLflow

---

## Project Structure

```text
111project/
├── artifacts/
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── brfss_final_training_report.txt
│   ├── brfss_test_confusion_matrix.png
│   ├── brfss_test_roc_curve.png
│   └── validation metric plots
├── artifact/
│   └── preprocessor.pkl
├── backend/
│   ├── app.py
│   ├── main.py
│   └── src/
├── frontend-deploy/
│   └── streamlit_app.py
├── src/
│   └── mlproject/
├── brfss_2024_eda_processed.csv
├── train_brfss_option_b.py
├── app.py
├── main.py
├── streamlit_app.py
└── README.md
```

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/msbisrat/111project.git
cd 111project
```

### 2. Switch to the project branch

```bash
git switch final-final-zip
```

### 3. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Create a `.env` file

```env
GROQ_API_KEY=your_groq_api_key
```

### 6. Train the BRFSS model

```bash
python train_brfss_option_b.py
```

This creates or updates the model, preprocessor, report, and evaluation plots in the `artifacts/` folder.

### 7. Run the app

Depending on which app entry point you are using:

```bash
streamlit run streamlit_app.py
```

or:

```bash
streamlit run frontend-deploy/streamlit_app.py
```

If using the FastAPI backend separately:

```bash
uvicorn backend.app:app --reload
```

---

## Important Implementation Notes

The repository contains both the original UCI-style clinical heart disease pipeline and the newer BRFSS-based model work.

The current BRFSS training script uses BRFSS survey features such as demographics, lifestyle behavior, and comorbidities. Some app UI fields still look like the older UCI clinical dataset inputs, such as chest pain type, cholesterol, ECG results, ST depression, and thalassemia.

Because of this, one important future improvement is to align the frontend form fields fully with the 24 BRFSS features used by the final model.

---

## Limitations

- The dataset is imbalanced.
- The final model improves recall for heart disease cases but has low precision.
- More false positives are expected because the model is optimized to catch more positive cases.
- Some UI inputs may not map directly to the current BRFSS model features.
- Survey data may contain self-reporting errors.
- The AI assistant can generate helpful summaries, but it is not a medical authority.
- The project is not a diagnostic tool.



