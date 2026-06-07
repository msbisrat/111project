import os
import tempfile
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from src.mlproject.risk_adjustment import BRFSSRiskAdjuster


CSV_PATH = "brfss_survey_data_processed.csv"
TARGET_COL = "HadHeartDisease"
ARTIFACT_DIR = "artifacts"


def yes_no_to_code(value):
    """Convert yes/no style BRFSS values to 1/2 codes used by risk_adjustment.py."""
    if pd.isna(value):
        return None

    text = str(value).strip().lower()

    if text in ["yes", "1", "1.0", "true"]:
        return 1
    if text in ["no", "2", "2.0", "false"]:
        return 2

    try:
        return int(float(value))
    except Exception:
        return None


def numeric_or_none(value):
    if pd.isna(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def row_to_profile(row):
    """Convert one BRFSS CSV row into the profile format expected by BRFSSRiskAdjuster.adjust()."""
    return {
        "sex": numeric_or_none(row.get("Sex")),
        "age_category": numeric_or_none(row.get("AgeCategory")),
        "education": numeric_or_none(row.get("Education")),
        "income": numeric_or_none(row.get("Income")),
        "employment_status": numeric_or_none(row.get("EmploymentStatus")),
        "marital_status": numeric_or_none(row.get("MaritalStatus")),
        "home_ownership": numeric_or_none(row.get("HomeOwnership")),

        "general_health": numeric_or_none(row.get("GeneralHealth")),
        "good_or_better_health": numeric_or_none(row.get("GoodOrBetterHealth")),
        "height": numeric_or_none(row.get("Height")),
        "weight": numeric_or_none(row.get("Weight")),

        "smoked_100_cigarettes": yes_no_to_code(row.get("Smoked100Cigarettes")),
        "smoker_status": numeric_or_none(row.get("SmokerStatus")),
        "ecigarette_usage": numeric_or_none(row.get("ECigaretteUsage")),
        "smokeless_tobacco_use": numeric_or_none(row.get("SmokelessTobaccoUse")),

        "alcohol_days": numeric_or_none(row.get("AlcoholDays")),
        "alcohol_drinkers": numeric_or_none(row.get("AlcoholDrinkers")),

        "physical_activities": numeric_or_none(row.get("PhysicalActivities")),

        "had_angina": yes_no_to_code(row.get("HadAngina")),
        "had_depressive_disorder": yes_no_to_code(row.get("HadDepressiveDisorder")),
        "had_diabetes": yes_no_to_code(row.get("HadDiabetes")),
        "had_kidney_disease": yes_no_to_code(row.get("HadKidneyDisease")),
        "had_stroke": yes_no_to_code(row.get("HadStroke")),
        "had_copd": yes_no_to_code(row.get("HadCOPD")),
        "had_arthritis": yes_no_to_code(row.get("HadArthritis")),
    }


def target_to_binary(value):
    text = str(value).strip().lower()
    if text in ["yes", "1", "1.0", "true"]:
        return 1
    if text in ["no", "2", "2.0", "false"]:
        return 0
    return None


def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"{CSV_PATH} not found. Put it in the project root, same level as app.py and streamlit_app.py."
        )

    df = pd.read_csv(CSV_PATH, low_memory=False)

    if TARGET_COL not in df.columns:
        raise ValueError(f"{TARGET_COL} column not found in {CSV_PATH}")

    df["_target"] = df[TARGET_COL].apply(target_to_binary)
    df = df.dropna(subset=["_target"]).copy()
    df["_target"] = df["_target"].astype(int)

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["_target"]
    )

    # Fit weights only on train split, then evaluate on test split.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        train_path = tmp.name
        train_df.drop(columns=["_target"]).to_csv(train_path, index=False)

    adjuster = BRFSSRiskAdjuster(train_path)
    weights = adjuster.fit()

    base_probability = weights.get("baseline_cvd_rate", train_df["_target"].mean())

    y_true = []
    y_prob = []

    for _, row in test_df.iterrows():
        profile = row_to_profile(row)
        result = adjuster.adjust(base_probability=base_probability, profile=profile)

        y_true.append(int(row["_target"]))
        y_prob.append(float(result["adjusted_probability"]))

    # threshold can be changed, but 0.5 is the simple default
    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    report_path = os.path.join(ARTIFACT_DIR, "brfss_weight_adjuster_evaluation.txt")

    with open(report_path, "w") as f:
        f.write("BRFSS Weight-Adjustment Evaluation\n")
        f.write("=" * 70 + "\n\n")
        f.write("What this evaluates:\n")
        f.write("This evaluates the BRFSS relative-risk weighting stage only.\n")
        f.write("It does not evaluate the full UCI + BRFSS app together because the UCI and BRFSS datasets are not paired by the same users.\n\n")

        f.write(f"CSV used: {CSV_PATH}\n")
        f.write(f"Train rows for weights: {len(train_df)}\n")
        f.write(f"Test rows for evaluation: {len(test_df)}\n")
        f.write(f"Base probability used: {base_probability:.4f}\n\n")

        f.write("Metrics using threshold 0.50\n")
        f.write("-" * 70 + "\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1-score:  {f1:.4f}\n")
        f.write(f"ROC AUC:   {roc_auc:.4f}\n\n")

        f.write("Class labels:\n")
        f.write("Class 0 = No Heart Disease\n")
        f.write("Class 1 = Had Heart Disease\n\n")

        f.write("Classification Report\n")
        f.write("-" * 70 + "\n")
        f.write(classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["No Heart Disease", "Had Heart Disease"],
            zero_division=0
        ))

        f.write("\nConfusion Matrix\n")
        f.write("-" * 70 + "\n")
        f.write(str(cm))
        f.write("\n\n")

        f.write("Learned BRFSS Relative-Risk Weights\n")
        f.write("-" * 70 + "\n")
        for key, value in weights.items():
            f.write(f"{key}: {value}\n")

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Heart Disease", "Had Heart Disease"]
    )
    disp.plot(values_format="d")
    plt.title("BRFSS Weight Adjuster Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "brfss_weight_adjuster_confusion_matrix.png"), dpi=200)
    plt.close()

    print("\nBRFSS Weight-Adjustment Evaluation")
    print("----------------------------------")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")
    print("\nConfusion matrix:")
    print(cm)

    print(f"\nSaved report: {report_path}")
    print("Saved graph: artifacts/brfss_weight_adjuster_confusion_matrix.png")


if __name__ == "__main__":
    main()