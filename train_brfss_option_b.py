"""
train_brfss_option_b.py

Option B: fully retrain the heart disease prediction model using BRFSS features only.

Input CSV:
    brfss_2024_eda_processed.csv

Output:
    artifacts/model.pkl
    artifact/preprocessor.pkl
    artifacts/brfss_model_comparison.csv
    artifacts/brfss_test_classification_report.txt

Run from project root:
    python3 train_brfss_option_b.py
"""

from pathlib import Path
import pickle

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


DATA_PATH = Path("brfss_2024_eda_processed.csv")
TARGET_COL = "HadHeartDisease"

# Option B BRFSS-only features.
# We exclude HadHeartAttack and HadAngina because they are too close to HadHeartDisease
# and could create target leakage.
FEATURE_COLS = [
    "Sex",
    "AgeCategory",
    "Education",
    "Income",
    "EmploymentStatus",
    "MaritalStatus",
    "HomeOwnership",
    "GeneralHealth",
    "GoodOrBetterHealth",
    "LastCheckup",
    "Height",
    "Weight",
    "Smoked100Cigarettes",
    "SmokerStatus",
    "ECigaretteUsage",
    "SmokelessTobaccoUse",
    "AlcoholDays",
    "PhysicalActivities",
    "HadDiabetes",
    "HadKidneyDisease",
    "HadStroke",
    "HadCOPD",
    "HadDepressiveDisorder",
    "HadArthritis",
]

NUMERIC_FEATURES = [
    "Sex",
    "AgeCategory",
    "Education",
    "Income",
    "EmploymentStatus",
    "MaritalStatus",
    "HomeOwnership",
    "GeneralHealth",
    "GoodOrBetterHealth",
    "LastCheckup",
    "Height",
    "Weight",
    "SmokerStatus",
    "ECigaretteUsage",
    "SmokelessTobaccoUse",
    "AlcoholDays",
    "PhysicalActivities",
]

CATEGORICAL_FEATURES = [
    "Smoked100Cigarettes",
    "HadDiabetes",
    "HadKidneyDisease",
    "HadStroke",
    "HadCOPD",
    "HadDepressiveDisorder",
    "HadArthritis",
]


def convert_target(series: pd.Series) -> pd.Series:
    """Convert Yes/No target to binary labels."""
    return series.map({"Yes": 1, "No": 0})


def get_positive_probability(model, X):
    """Return positive-class probability or a fallback score."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def evaluate_model(name, model, X_val, y_val):
    preds = model.predict(X_val)
    probs = get_positive_probability(model, X_val)

    result = {
        "model": name,
        "accuracy": accuracy_score(y_val, preds),
        "precision_class_1": precision_score(y_val, preds, zero_division=0),
        "recall_class_1": recall_score(y_val, preds, zero_division=0),
        "f1_class_1": f1_score(y_val, preds, zero_division=0),
    }

    if probs is not None:
        result["roc_auc"] = roc_auc_score(y_val, probs)
    else:
        result["roc_auc"] = None

    return result


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {DATA_PATH}. Put brfss_2024_eda_processed.csv in the project root folder."
        )

    df = pd.read_csv(DATA_PATH)

    missing_cols = [col for col in [TARGET_COL] + FEATURE_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in CSV: {missing_cols}")

    df = df[FEATURE_COLS + [TARGET_COL]].copy()
    df[TARGET_COL] = convert_target(df[TARGET_COL])
    df = df.dropna(subset=[TARGET_COL])
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    print("Dataset shape:", df.shape)
    print("Target distribution:")
    print(y.value_counts(normalize=False))
    print(y.value_counts(normalize=True))

    # 60% train, 20% validation, 20% test.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        random_state=42,
        stratify=y_train_val,
    )

    print("\nSplit sizes:")
    print("Train:", X_train.shape, "Validation:", X_val.shape, "Test:", X_test.shape)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)

    # Simple model comparison for presentation.
    # Hyperparameters are intentionally small/simple for local runtime.
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=50,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=50,
            max_depth=12,
            min_samples_leaf=30,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
            verbose=1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=50,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }

    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_t, y_train)
        val_result = evaluate_model(name, model, X_val_t, y_val)
        results.append(val_result)
        print("Validation result:", val_result)

    results_df = pd.DataFrame(results)
    print("\nValidation comparison:")
    print(results_df.sort_values(by="f1_class_1", ascending=False))

    # Choose best model by class-1 F1-score.
    best_row = results_df.sort_values(by="f1_class_1", ascending=False).iloc[0]
    best_name = best_row["model"]
    best_model = models[best_name]

    print(f"\nBest model by validation F1 for heart disease class: {best_name}")

    test_preds = best_model.predict(X_test_t)
    test_probs = get_positive_probability(best_model, X_test_t)

    print("\nFinal Test Accuracy:", accuracy_score(y_test, test_preds))
    print("\nFinal Test Confusion Matrix:")
    print(confusion_matrix(y_test, test_preds))
    print("\nFinal Test Classification Report:")
    test_report = classification_report(y_test, test_preds)
    print(test_report)

    if test_probs is not None:
        print("\nFinal Test ROC AUC:", roc_auc_score(y_test, test_probs))

    Path("artifacts").mkdir(exist_ok=True)
    Path("artifact").mkdir(exist_ok=True)

    results_df.to_csv("artifacts/brfss_model_comparison.csv", index=False)

    with open("artifacts/brfss_test_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Best model: {best_name}\n\n")
        f.write("Validation comparison:\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\nFinal Test Confusion Matrix:\n")
        f.write(str(confusion_matrix(y_test, test_preds)))
        f.write("\n\nFinal Test Classification Report:\n")
        f.write(test_report)
        if test_probs is not None:
            f.write(f"\nFinal Test ROC AUC: {roc_auc_score(y_test, test_probs)}\n")

    # Save best BRFSS-only model and its preprocessor.
    # PredictPipeline can keep loading these same two filenames.
    with open("artifacts/model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    with open("artifact/preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    print("\nSaved best BRFSS-only model to artifacts/model.pkl")
    print("Saved BRFSS preprocessor to artifact/preprocessor.pkl")
    print("Saved model comparison to artifacts/brfss_model_comparison.csv")


if __name__ == "__main__":
    main()
