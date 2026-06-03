"""
train_brfss_option_b_final_no_threshold_txt_png.py

Final BRFSS Option B training script.

Purpose:
- Use normal untuned baseline models first, then compare them with tuned models.
- Use tuned models after that, with imbalance handling and stronger hyperparameters.
- Select the best model honestly based on validation F1-score for the heart disease class.
- No threshold tuning. All models use their normal predict() behavior.
- Save only TXT and PNG outputs, plus model/preprocessor pickle files.
- Print progress during training.

Run:
    python3 train_brfss_option_b_final_no_threshold_txt_png.py

Required CSV in project root:
    brfss_2024_eda_processed.csv

Outputs:
    artifacts/model.pkl
    artifact/preprocessor.pkl
    artifacts/brfss_final_training_report.txt
    artifacts/brfss_validation_f1_comparison.png
    artifacts/brfss_validation_precision_comparison.png
    artifacts/brfss_validation_recall_comparison.png
    artifacts/brfss_validation_roc_auc_comparison.png
    artifacts/brfss_best_worst_before_after.png
    artifacts/brfss_test_confusion_matrix.png
    artifacts/brfss_test_roc_curve.png
"""

from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    ConfusionMatrixDisplay,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight


DATA_PATH = Path("brfss_2024_eda_processed.csv")
TARGET_COL = "HadHeartDisease"

ARTIFACT_DIR = Path("artifacts")
PREPROCESSOR_DIR = Path("artifact")

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
    """Convert target from Yes/No to 1/0."""
    return series.map({"Yes": 1, "No": 0})


def get_positive_probability(model, x_data):
    """Return probability/score for class 1 if the model supports it."""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x_data)
        if probs.shape[1] == 2:
            return probs[:, 1]
        # If model sees one class only, return zeros as fallback.
        return np.zeros(x_data.shape[0])

    if hasattr(model, "decision_function"):
        scores = model.decision_function(x_data)
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)

    return None


def evaluate_predictions(y_true, preds, probs):
    """Compute evaluation metrics."""
    result = {
        "accuracy": accuracy_score(y_true, preds),
        "precision_class_1": precision_score(y_true, preds, zero_division=0),
        "recall_class_1": recall_score(y_true, preds, zero_division=0),
        "f1_class_1": f1_score(y_true, preds, zero_division=0),
    }

    if probs is not None:
        try:
            result["roc_auc"] = roc_auc_score(y_true, probs)
        except ValueError:
            result["roc_auc"] = 0.5
    else:
        result["roc_auc"] = None

    return result


def evaluate_model(group_name, model_name, params, model, x_val, y_val, train_seconds):
    """Evaluate model on validation set using default predict(), no threshold tuning."""
    preds = model.predict(x_val)
    probs = get_positive_probability(model, x_val)
    metrics = evaluate_predictions(y_val, preds, probs)

    return {
        "group": group_name,
        "model": model_name,
        "params": params,
        "train_seconds": train_seconds,
        **metrics,
    }


def selection_key(row):
    """
    Select based mainly on class-1 F1-score, then recall, then precision.
    This avoids selecting only by accuracy on an imbalanced dataset.
    """
    return (
        row["f1_class_1"],
        row["recall_class_1"],
        row["precision_class_1"],
        row["roc_auc"] if row["roc_auc"] is not None else 0,
    )


def get_baseline_configs():
    """
    Before tuning baseline:
    Use the same main model families as the tuned stage, but with simple/default settings
    and no imbalance handling.

    This is a fair baseline because it compares normal untuned models against tuned models:
    - no class_weight="balanced"
    - no balanced sample_weight
    - no threshold tuning
    """
    return [
        {
            "group": "Before tuning baseline",
            "name": "Logistic Regression",
            "params": "default style; no class_weight",
            "use_balanced_sample_weight": False,
            "model": LogisticRegression(max_iter=1000, random_state=42),
        },
        {
            "group": "Before tuning baseline",
            "name": "Decision Tree",
            "params": "default tree; no class_weight",
            "use_balanced_sample_weight": False,
            "model": DecisionTreeClassifier(random_state=42),
        },
        {
            "group": "Before tuning baseline",
            "name": "Random Forest",
            "params": "default RF; n_estimators=100; no class_weight",
            "use_balanced_sample_weight": False,
            "model": RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                verbose=1,
            ),
        },
        {
            "group": "Before tuning baseline",
            "name": "Gradient Boosting",
            "params": "default gradient boosting; no sample_weight",
            "use_balanced_sample_weight": False,
            "model": GradientBoostingClassifier(
                random_state=42,
                verbose=1,
            ),
        },
    ]


def get_tuned_configs():
    """
    After tuning:
    Stronger models with imbalance handling and tuned hyperparameters.
    No threshold tuning is used.
    """
    return [
        {
            "group": "After tuning",
            "name": "Logistic Regression",
            "params": "C=1.0; class_weight=balanced",
            "use_balanced_sample_weight": False,
            "model": LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        },
        {
            "group": "After tuning",
            "name": "Decision Tree",
            "params": "max_depth=8; min_samples_leaf=50; class_weight=balanced",
            "use_balanced_sample_weight": False,
            "model": DecisionTreeClassifier(
                max_depth=8,
                min_samples_leaf=50,
                class_weight="balanced",
                random_state=42,
            ),
        },
        {
            "group": "After tuning",
            "name": "Decision Tree",
            "params": "max_depth=12; min_samples_leaf=80; class_weight=balanced",
            "use_balanced_sample_weight": False,
            "model": DecisionTreeClassifier(
                max_depth=12,
                min_samples_leaf=80,
                class_weight="balanced",
                random_state=42,
            ),
        },
        {
            "group": "After tuning",
            "name": "Random Forest",
            "params": "n_estimators=100; max_depth=12; min_samples_leaf=30; class_weight=balanced",
            "use_balanced_sample_weight": False,
            "model": RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_leaf=30,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
                verbose=1,
            ),
        },
        {
            "group": "After tuning",
            "name": "Random Forest",
            "params": "n_estimators=200; max_depth=12; min_samples_leaf=30; class_weight=balanced",
            "use_balanced_sample_weight": False,
            "model": RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=30,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
                verbose=1,
            ),
        },
        {
            "group": "After tuning",
            "name": "Random Forest",
            "params": "n_estimators=100; max_depth=16; min_samples_leaf=20; class_weight=balanced",
            "use_balanced_sample_weight": False,
            "model": RandomForestClassifier(
                n_estimators=100,
                max_depth=16,
                min_samples_leaf=20,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
                verbose=1,
            ),
        },
        {
            "group": "After tuning",
            "name": "Gradient Boosting",
            "params": "n_estimators=100; learning_rate=0.05; max_depth=3; balanced sample_weight",
            "use_balanced_sample_weight": True,
            "model": GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                random_state=42,
                verbose=1,
            ),
        },
        {
            "group": "After tuning",
            "name": "Gradient Boosting",
            "params": "n_estimators=200; learning_rate=0.05; max_depth=3; balanced sample_weight",
            "use_balanced_sample_weight": True,
            "model": GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=42,
                verbose=1,
            ),
        },
        {
            "group": "After tuning",
            "name": "Gradient Boosting",
            "params": "n_estimators=100; learning_rate=0.10; max_depth=3; balanced sample_weight",
            "use_balanced_sample_weight": True,
            "model": GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.10,
                max_depth=3,
                random_state=42,
                verbose=1,
            ),
        },
    ]


def result_to_text(row):
    """Format one result row for report."""
    return (
        f"Group: {row['group']}\n"
        f"Model: {row['model']}\n"
        f"Params: {row['params']}\n"
        f"Accuracy: {row['accuracy']:.4f}\n"
        f"Precision class 1: {row['precision_class_1']:.4f}\n"
        f"Recall class 1: {row['recall_class_1']:.4f}\n"
        f"F1 class 1: {row['f1_class_1']:.4f}\n"
        f"ROC AUC: {row['roc_auc']:.4f}\n"
        f"Training seconds: {row['train_seconds']:.2f}\n"
    )


def save_metric_chart(results_df, metric, title, filename):
    labels = [f"{r.group}\n{r.model}" for r in results_df.itertuples()]
    values = results_df[metric].to_numpy()

    plt.figure(figsize=(15, 8))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), labels, rotation=75, ha="right")
    plt.ylabel(metric)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / filename, dpi=200)
    plt.close()


def save_best_worst_chart(best_before, worst_before, best_after, worst_after):
    labels = ["Best Baseline", "Worst Baseline", "Best Tuned", "Worst Tuned"]

    f1_values = [
        best_before["f1_class_1"],
        worst_before["f1_class_1"],
        best_after["f1_class_1"],
        worst_after["f1_class_1"],
    ]
    precision_values = [
        best_before["precision_class_1"],
        worst_before["precision_class_1"],
        best_after["precision_class_1"],
        worst_after["precision_class_1"],
    ]
    recall_values = [
        best_before["recall_class_1"],
        worst_before["recall_class_1"],
        best_after["recall_class_1"],
        worst_after["recall_class_1"],
    ]

    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(11, 6))
    plt.bar(x - width, f1_values, width, label="F1")
    plt.bar(x, precision_values, width, label="Precision")
    plt.bar(x + width, recall_values, width, label="Recall")
    plt.xticks(x, labels)
    plt.ylabel("Score")
    plt.title("Best/Worst Baseline vs Best/Worst Tuned Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "brfss_best_worst_before_after.png", dpi=200)
    plt.close()


def save_confusion_matrix_plot(cm):
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Heart Disease", "Had Heart Disease"],
    )
    display.plot(values_format="d")
    plt.title("Final Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "brfss_test_confusion_matrix.png", dpi=200)
    plt.close()


def save_roc_curve_plot(y_true, probs, roc_auc_value):
    if probs is None:
        return

    fpr, tpr, _ = roc_curve(y_true, probs)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_value:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate / Recall")
    plt.title("Final Test ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "brfss_test_roc_curve.png", dpi=200)
    plt.close()


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {DATA_PATH}. Put brfss_2024_eda_processed.csv in the project root folder."
        )

    ARTIFACT_DIR.mkdir(exist_ok=True)
    PREPROCESSOR_DIR.mkdir(exist_ok=True)

    total_start = time.perf_counter()

    print("\nLoading BRFSS data...")
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
    print("\nTarget distribution:")
    print(y.value_counts())
    print("\nTarget distribution proportion:")
    print(y.value_counts(normalize=True))

    print("\nSplitting data: 60% train, 20% validation, 20% test...")
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

    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    print("\nBuilding preprocessing pipeline...")
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

    print("\nFitting preprocessing on training data...")
    X_train_t = preprocessor.fit_transform(X_train)
    print("Transforming validation data...")
    X_val_t = preprocessor.transform(X_val)
    print("Transforming test data...")
    X_test_t = preprocessor.transform(X_test)

    balanced_sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    configs = get_baseline_configs() + get_tuned_configs()
    results = []
    trained_models = []

    print(f"\nStarting model training. Total runs: {len(configs)}")

    for i, cfg in enumerate(configs, start=1):
        group_name = cfg["group"]
        model_name = cfg["name"]
        params = cfg["params"]
        model = cfg["model"]
        use_balanced_sample_weight = cfg["use_balanced_sample_weight"]

        print("\n" + "=" * 80)
        print(f"Progress: {i}/{len(configs)}")
        print(f"Stage: {group_name}")
        print(f"Model: {model_name}")
        print(f"Params: {params}")
        print("Training started...")

        start = time.perf_counter()

        if use_balanced_sample_weight:
            model.fit(X_train_t, y_train, sample_weight=balanced_sample_weight)
        else:
            model.fit(X_train_t, y_train)

        train_seconds = time.perf_counter() - start
        print(f"Training finished in {train_seconds:.2f} seconds.")
        print("Evaluating on validation set...")

        result = evaluate_model(
            group_name,
            model_name,
            params,
            model,
            X_val_t,
            y_val,
            train_seconds,
        )

        results.append(result)
        trained_models.append((group_name, model_name, params, model))

        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"Precision class 1: {result['precision_class_1']:.4f}")
        print(f"Recall class 1: {result['recall_class_1']:.4f}")
        print(f"F1 class 1: {result['f1_class_1']:.4f}")
        print(f"ROC AUC: {result['roc_auc']:.4f}")
        print(f"Total elapsed time: {(time.perf_counter() - total_start) / 60:.2f} minutes")

    results_df = pd.DataFrame(results)

    # Sort for reporting. Main priority is F1, then recall, then precision.
    results_df_sorted = results_df.sort_values(
        by=["f1_class_1", "recall_class_1", "precision_class_1", "roc_auc"],
        ascending=False,
    )

    baseline_df = results_df[results_df["group"] == "Before tuning baseline"]
    tuned_df = results_df[results_df["group"] == "After tuning"]

    best_before = baseline_df.sort_values(
        by=["f1_class_1", "recall_class_1", "precision_class_1", "roc_auc"],
        ascending=False,
    ).iloc[0]

    worst_before = baseline_df.sort_values(
        by=["f1_class_1", "recall_class_1", "precision_class_1", "roc_auc"],
        ascending=True,
    ).iloc[0]

    best_after = tuned_df.sort_values(
        by=["f1_class_1", "recall_class_1", "precision_class_1", "roc_auc"],
        ascending=False,
    ).iloc[0]

    worst_after = tuned_df.sort_values(
        by=["f1_class_1", "recall_class_1", "precision_class_1", "roc_auc"],
        ascending=True,
    ).iloc[0]

    # Select final best model from tuned models only.
    # This makes sense because the final production model should come from the tuned stage.
    final_selected = best_after

    best_model = None
    for group_name, model_name, params, model in trained_models:
        if (
            group_name == final_selected["group"]
            and model_name == final_selected["model"]
            and params == final_selected["params"]
        ):
            best_model = model
            break

    if best_model is None:
        raise RuntimeError("Could not find selected best tuned model.")

    print("\n" + "=" * 80)
    print("Best BEFORE tuning baseline:")
    print(result_to_text(best_before))

    print("Worst BEFORE tuning baseline:")
    print(result_to_text(worst_before))

    print("Best AFTER tuning:")
    print(result_to_text(best_after))

    print("Worst AFTER tuning:")
    print(result_to_text(worst_after))

    print("Final selected model:")
    print(result_to_text(final_selected))

    print("\nEvaluating final selected tuned model on test set...")
    test_preds = best_model.predict(X_test_t)
    test_probs = get_positive_probability(best_model, X_test_t)

    test_metrics = evaluate_predictions(y_test, test_preds, test_probs)
    test_cm = confusion_matrix(y_test, test_preds)
    test_report = classification_report(y_test, test_preds)

    print("\nFinal Test Metrics:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision class 1: {test_metrics['precision_class_1']:.4f}")
    print(f"Recall class 1: {test_metrics['recall_class_1']:.4f}")
    print(f"F1 class 1: {test_metrics['f1_class_1']:.4f}")
    print(f"ROC AUC: {test_metrics['roc_auc']:.4f}")

    print("\nFinal Test Confusion Matrix:")
    print(test_cm)

    print("\nFinal Test Classification Report:")
    print(test_report)

    print("\nSaving TXT report...")
    with open(ARTIFACT_DIR / "brfss_final_training_report.txt", "w", encoding="utf-8") as f:
        f.write("BRFSS Option B Final Training Report\n")
        f.write("=" * 80)
        f.write("\n\n")

        f.write("Important note:\n")
        f.write(
            "The baseline stage uses simple/default models without imbalance handling. "
            "The tuned stage uses stronger models, class imbalance handling, and larger Random Forest sizes. "
            "The final selected model is chosen from the tuned stage based mainly on class-1 F1-score, "
            "then recall, then precision. ROC AUC is also reported.\n\n"
        )

        f.write("All validation results sorted by F1/Recall/Precision/ROC AUC:\n")
        f.write(results_df_sorted.to_string(index=False))
        f.write("\n\n")

        f.write("Best BEFORE tuning baseline:\n")
        f.write(result_to_text(best_before))
        f.write("\n")

        f.write("Worst BEFORE tuning baseline:\n")
        f.write(result_to_text(worst_before))
        f.write("\n")

        f.write("Best AFTER tuning:\n")
        f.write(result_to_text(best_after))
        f.write("\n")

        f.write("Worst AFTER tuning:\n")
        f.write(result_to_text(worst_after))
        f.write("\n")

        f.write("Final selected model:\n")
        f.write(result_to_text(final_selected))
        f.write("\n")

        f.write("Final Test Metrics:\n")
        f.write(f"Accuracy: {test_metrics['accuracy']}\n")
        f.write(f"Precision class 1: {test_metrics['precision_class_1']}\n")
        f.write(f"Recall class 1: {test_metrics['recall_class_1']}\n")
        f.write(f"F1 class 1: {test_metrics['f1_class_1']}\n")
        f.write(f"ROC AUC: {test_metrics['roc_auc']}\n")
        f.write("\nFinal Test Confusion Matrix:\n")
        f.write(str(test_cm))
        f.write("\n\nFinal Test Classification Report:\n")
        f.write(test_report)

    print("Saving graphs...")
    save_metric_chart(
        results_df_sorted,
        "f1_class_1",
        "Validation F1-score for Heart Disease Class",
        "brfss_validation_f1_comparison.png",
    )
    save_metric_chart(
        results_df_sorted,
        "precision_class_1",
        "Validation Precision for Heart Disease Class",
        "brfss_validation_precision_comparison.png",
    )
    save_metric_chart(
        results_df_sorted,
        "recall_class_1",
        "Validation Recall for Heart Disease Class",
        "brfss_validation_recall_comparison.png",
    )
    save_metric_chart(
        results_df_sorted,
        "roc_auc",
        "Validation ROC AUC Comparison",
        "brfss_validation_roc_auc_comparison.png",
    )
    save_best_worst_chart(best_before, worst_before, best_after, worst_after)
    save_confusion_matrix_plot(test_cm)
    save_roc_curve_plot(y_test, test_probs, test_metrics["roc_auc"])

    print("Saving final tuned model and preprocessor...")
    with open(ARTIFACT_DIR / "model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    with open(PREPROCESSOR_DIR / "preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    total_seconds = time.perf_counter() - total_start

    print("\n" + "=" * 80)
    print("DONE")
    print(f"Total runtime: {total_seconds / 60:.2f} minutes")
    print("\nSaved outputs:")
    print("- artifacts/model.pkl")
    print("- artifact/preprocessor.pkl")
    print("- artifacts/brfss_final_training_report.txt")
    print("- artifacts/brfss_validation_f1_comparison.png")
    print("- artifacts/brfss_validation_precision_comparison.png")
    print("- artifacts/brfss_validation_recall_comparison.png")
    print("- artifacts/brfss_validation_roc_auc_comparison.png")
    print("- artifacts/brfss_best_worst_before_after.png")
    print("- artifacts/brfss_test_confusion_matrix.png")
    print("- artifacts/brfss_test_roc_curve.png")


if __name__ == "__main__":
    main()
