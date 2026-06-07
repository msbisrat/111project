import os
import sys
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
# Avoid importing optional DB client at module import to prevent hard failures
try:
    import pymysql  # optional; only needed if reading from MySQL
except Exception:
    pymysql = None
import pickle
import numpy as np
load_dotenv()

host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db")



# Read SQL data
# This function connects to a MySQL database and reads data from a table named 'student'.

def read_sql_data():
    
    logging.info("Reading SQL database started")
    try:
        if pymysql is None:
            raise CustomException("pymysql not installed; install it or use CSV ingestion.")
        mydb = pymysql.connect(
            host = host,
            user = user,
            password = password,
            db = db
        )
        logging.info(f"Connection Established: {mydb}")
        df = pd.read_sql_query('Select * from heart' , mydb)
        return df
        
    
    except Exception as ex:
        raise CustomException(ex)
    
def save_object(file_path , obj)    :
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path , exist_ok = True)
        
        with open(file_path , 'wb') as file_obj:
            pickle.dump(obj , file_obj)
            
    except Exception as e:
        raise CustomException(e , sys)        
    

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.mlproject.exception import CustomException
import sys

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.preprocessing import LabelEncoder

def evaluate_model(X_train, y_train, X_test, y_test, models, param):
    try:
        import os
        import sys
        import pandas as pd
        import matplotlib.pyplot as plt

        from sklearn.base import clone
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import (
            accuracy_score,
            precision_recall_fscore_support,
            classification_report,
            confusion_matrix,
            ConfusionMatrixDisplay,
        )

        from src.mlproject.exception import CustomException

        os.makedirs("artifacts", exist_ok=True)

        rows = []
        full_reports = []
        tuned_report = {}
        tuned_predictions = {}
        tuned_best_params = {}

        for model_name, model in models.items():
            para = param[model_name]

            print("\n" + "=" * 80)
            print(f"Model: {model_name}")

            # ============================================================
            # 1. Baseline model: default model, no GridSearchCV
            # ============================================================
            print(f"\nTraining baseline {model_name}...")

            baseline_model = clone(model)
            baseline_model.fit(X_train, y_train)
            baseline_pred = baseline_model.predict(X_test)

            baseline_accuracy = accuracy_score(y_test, baseline_pred)

            base_p, base_r, base_f1, base_support = precision_recall_fscore_support(
                y_test,
                baseline_pred,
                labels=[0, 1],
                zero_division=0,
            )

            print(f"\nBaseline {model_name}")
            print(f"Accuracy: {baseline_accuracy:.4f}")
            print("Class-specific metrics:")
            print(f"Class 0 - No Heart Disease: precision={base_p[0]:.4f}, recall={base_r[0]:.4f}, F1={base_f1[0]:.4f}, support={base_support[0]}")
            print(f"Class 1 - Had Heart Disease: precision={base_p[1]:.4f}, recall={base_r[1]:.4f}, F1={base_f1[1]:.4f}, support={base_support[1]}")

            baseline_cm = confusion_matrix(y_test, baseline_pred, labels=[0, 1])
            print("Confusion matrix:")
            print(baseline_cm)

            rows.append({
                "model": model_name,
                "version": "Baseline",
                "accuracy": baseline_accuracy,
                "precision_class_0": base_p[0],
                "recall_class_0": base_r[0],
                "f1_class_0": base_f1[0],
                "support_class_0": base_support[0],
                "precision_class_1": base_p[1],
                "recall_class_1": base_r[1],
                "f1_class_1": base_f1[1],
                "support_class_1": base_support[1],
                "best_params": "Default parameters",
            })

            full_reports.append(
                f"\n{'=' * 80}\n"
                f"Baseline {model_name}\n"
                f"{'-' * 80}\n"
                f"Accuracy: {baseline_accuracy:.4f}\n\n"
                f"{classification_report(y_test, baseline_pred, labels=[0, 1], target_names=['No Heart Disease', 'Had Heart Disease'], zero_division=0)}\n"
                f"Confusion matrix:\n{baseline_cm}\n"
            )

            # ============================================================
            # 2. Tuned model: Millen's original GridSearchCV tuning
            # ============================================================
            print(f"\nTuning {model_name} with GridSearchCV...")

            gs = GridSearchCV(clone(model), para, cv=3)
            gs.fit(X_train, y_train)

            tuned_model = gs.best_estimator_
            tuned_pred = tuned_model.predict(X_test)

            tuned_accuracy = accuracy_score(y_test, tuned_pred)

            tuned_p, tuned_r, tuned_f1, tuned_support = precision_recall_fscore_support(
                y_test,
                tuned_pred,
                labels=[0, 1],
                zero_division=0,
            )

            print(f"\nTuned {model_name}")
            print(f"Best parameters: {gs.best_params_}")
            print(f"Accuracy: {tuned_accuracy:.4f}")
            print("Class-specific metrics:")
            print(f"Class 0 - No Heart Disease: precision={tuned_p[0]:.4f}, recall={tuned_r[0]:.4f}, F1={tuned_f1[0]:.4f}, support={tuned_support[0]}")
            print(f"Class 1 - Had Heart Disease: precision={tuned_p[1]:.4f}, recall={tuned_r[1]:.4f}, F1={tuned_f1[1]:.4f}, support={tuned_support[1]}")

            tuned_cm = confusion_matrix(y_test, tuned_pred, labels=[0, 1])
            print("Confusion matrix:")
            print(tuned_cm)

            rows.append({
                "model": model_name,
                "version": "Tuned",
                "accuracy": tuned_accuracy,
                "precision_class_0": tuned_p[0],
                "recall_class_0": tuned_r[0],
                "f1_class_0": tuned_f1[0],
                "support_class_0": tuned_support[0],
                "precision_class_1": tuned_p[1],
                "recall_class_1": tuned_r[1],
                "f1_class_1": tuned_f1[1],
                "support_class_1": tuned_support[1],
                "best_params": str(gs.best_params_),
            })

            full_reports.append(
                f"\n{'=' * 80}\n"
                f"Tuned {model_name}\n"
                f"{'-' * 80}\n"
                f"Best parameters: {gs.best_params_}\n"
                f"Accuracy: {tuned_accuracy:.4f}\n\n"
                f"{classification_report(y_test, tuned_pred, labels=[0, 1], target_names=['No Heart Disease', 'Had Heart Disease'], zero_division=0)}\n"
                f"Confusion matrix:\n{tuned_cm}\n"
            )

            # Keep Millen's original pipeline behavior:
            # model_trainer.py will save the selected tuned model.
            models[model_name] = tuned_model

            # Best model is selected by tuned accuracy, same as Millen's original logic.
            tuned_report[model_name] = tuned_accuracy
            tuned_predictions[model_name] = tuned_pred
            tuned_best_params[model_name] = gs.best_params_

        results_df = pd.DataFrame(rows)

        baseline_df = results_df[results_df["version"] == "Baseline"].copy()
        tuned_df = results_df[results_df["version"] == "Tuned"].copy()

        best_baseline = baseline_df.sort_values("accuracy", ascending=False).iloc[0]
        worst_baseline = baseline_df.sort_values("accuracy", ascending=True).iloc[0]

        best_tuned = tuned_df.sort_values("accuracy", ascending=False).iloc[0]
        worst_tuned = tuned_df.sort_values("accuracy", ascending=True).iloc[0]

        sorted_tuned = sorted(tuned_report.items(), key=lambda x: x[1], reverse=True)
        best_tuned_model_name = sorted_tuned[0][0]
        best_tuned_pred = tuned_predictions[best_tuned_model_name]
        best_tuned_cm = confusion_matrix(y_test, best_tuned_pred, labels=[0, 1])

        print("\n" + "=" * 80)
        print("Tuned Model Ranking by Accuracy")
        print("=" * 80)
        for model_name, score in sorted_tuned:
            print(f"{model_name}: {score:.4f}")

        print("\nBest/Worst Summary")
        print("-" * 80)
        print(f"Best baseline: {best_baseline['model']} | Accuracy: {best_baseline['accuracy']:.4f}")
        print(f"Worst baseline: {worst_baseline['model']} | Accuracy: {worst_baseline['accuracy']:.4f}")
        print(f"Best tuned: {best_tuned['model']} | Accuracy: {best_tuned['accuracy']:.4f}")
        print(f"Worst tuned: {worst_tuned['model']} | Accuracy: {worst_tuned['accuracy']:.4f}")
        print(f"\nFinal selected tuned model: {best_tuned_model_name}")
        print(f"Best tuned model confusion matrix:")
        print(best_tuned_cm)

        # ============================================================
        # Save TXT report
        # ============================================================
        txt_path = "artifacts/millen_baseline_vs_tuned_evaluation.txt"

        with open(txt_path, "w") as f:
            f.write("Millen Pipeline Baseline vs Tuned Evaluation\n")
            f.write("=" * 80 + "\n\n")

            f.write("Dataset used by Millen training pipeline:\n")
            f.write("notebook/data/cleaned_data.csv\n\n")

            f.write("Important note:\n")
            f.write("This cleaned clinical dataset is small and very balanced in the test split.\n")
            f.write("In this run, class 0 and class 1 have almost the same support, so baseline scores are already high.\n")
            f.write("This is different from the larger BRFSS dataset, which is more imbalanced.\n\n")

            f.write("Class labels:\n")
            f.write("Class 0 = No Heart Disease\n")
            f.write("Class 1 = Had Heart Disease\n\n")

            f.write("Model selection rule:\n")
            f.write("The final tuned model is selected by highest tuned accuracy, following Millen's original pipeline logic.\n\n")

            f.write("All Model Results\n")
            f.write("-" * 80 + "\n")
            f.write(results_df.to_string(index=False))
            f.write("\n\n")

            f.write("Tuned Model Ranking by Accuracy\n")
            f.write("-" * 80 + "\n")
            for model_name, score in sorted_tuned:
                f.write(f"{model_name}: {score:.4f}\n")
            f.write("\n")

            f.write("Best/Worst Summary\n")
            f.write("-" * 80 + "\n")
            f.write(f"Best baseline: {best_baseline['model']} | Accuracy: {best_baseline['accuracy']:.4f}\n")
            f.write(f"Worst baseline: {worst_baseline['model']} | Accuracy: {worst_baseline['accuracy']:.4f}\n")
            f.write(f"Best tuned: {best_tuned['model']} | Accuracy: {best_tuned['accuracy']:.4f}\n")
            f.write(f"Worst tuned: {worst_tuned['model']} | Accuracy: {worst_tuned['accuracy']:.4f}\n\n")

            f.write(f"Final selected tuned model: {best_tuned_model_name}\n")
            f.write(f"Best parameters: {tuned_best_params[best_tuned_model_name]}\n")
            f.write("Best tuned model confusion matrix:\n")
            f.write(str(best_tuned_cm))
            f.write("\n\n")

            f.write("Detailed Class Reports\n")
            f.write("=" * 80 + "\n")
            for report_text in full_reports:
                f.write(report_text)

        print(f"\nSaved TXT evaluation report to: {txt_path}")

        # ============================================================
        # Save PNG graph: Baseline vs Tuned Accuracy
        # ============================================================
        pivot_accuracy = results_df.pivot(index="model", columns="version", values="accuracy")
        pivot_accuracy.plot(kind="bar", figsize=(12, 6))
        plt.title("Baseline vs Tuned Model Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Model")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig("artifacts/millen_baseline_vs_tuned_accuracy.png", dpi=200)
        plt.close()

        # ============================================================
        # Save PNG graph: Baseline vs Tuned Class 1 F1
        # ============================================================
        pivot_f1_class1 = results_df.pivot(index="model", columns="version", values="f1_class_1")
        pivot_f1_class1.plot(kind="bar", figsize=(12, 6))
        plt.title("Baseline vs Tuned F1-score for Had Heart Disease Class")
        plt.ylabel("F1-score class 1")
        plt.xlabel("Model")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig("artifacts/millen_baseline_vs_tuned_f1_class1.png", dpi=200)
        plt.close()

        # ============================================================
        # Save PNG graph: Tuned Class 0 vs Class 1 F1
        # ============================================================
        tuned_graph_df = tuned_df.set_index("model")[["f1_class_0", "f1_class_1"]]
        tuned_graph_df.plot(kind="bar", figsize=(12, 6))
        plt.title("Tuned Models: Class 0 vs Class 1 F1-score")
        plt.ylabel("F1-score")
        plt.xlabel("Model")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig("artifacts/millen_tuned_class0_vs_class1_f1.png", dpi=200)
        plt.close()

        # ============================================================
        # Save PNG graph: Confusion Matrix for Best Tuned Model
        # ============================================================
        disp = ConfusionMatrixDisplay(
            confusion_matrix=best_tuned_cm,
            display_labels=["No Heart Disease", "Had Heart Disease"]
        )
        disp.plot(values_format="d")
        plt.title(f"Confusion Matrix: Best Tuned Model ({best_tuned_model_name})")
        plt.tight_layout()
        plt.savefig("artifacts/millen_best_tuned_confusion_matrix.png", dpi=200)
        plt.close()

        print("Saved PNG graphs:")
        print("artifacts/millen_baseline_vs_tuned_accuracy.png")
        print("artifacts/millen_baseline_vs_tuned_f1_class1.png")
        print("artifacts/millen_tuned_class0_vs_class1_f1.png")
        print("artifacts/millen_best_tuned_confusion_matrix.png")

        return tuned_report

    except Exception as e:
        raise CustomException(e, sys)