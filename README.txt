# Option B BRFSS-only update

Use these files if the team chooses Option B:

**Option B = fully retrain prediction model using BRFSS features only.**
The app should not require old UCI medical fields like thalassemia, ECG, major vessels, ST depression, or max heart rate.

## Files in this zip

- `train_brfss_option_b.py`
- `app.py`
- `streamlit_app.py`

## Required CSV

Put this file in the project root folder:

```text
brfss_2024_eda_processed.csv
```

## Run training

```bash
cd /Users/sarah/Downloads/111project-main
source .venv/bin/activate
python3 train_brfss_option_b.py
```

The script saves:

```text
artifacts/model.pkl
artifact/preprocessor.pkl
artifacts/brfss_model_comparison.csv
artifacts/brfss_test_classification_report.txt
```

## Run the app

Terminal 1:

```bash
uvicorn app:app --reload
```

Terminal 2:

```bash
streamlit run streamlit_app.py
```

Open:

```text
http://localhost:8501
```
