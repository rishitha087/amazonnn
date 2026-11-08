import os
import sys
import json
import yaml
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def evaluate_model():
    print("=" * 60)
    print("STAGE 6: COMPREHENSIVE MODEL EVALUATION & ERROR ANALYSIS")
    print("=" * 60)
    
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]
    X_test = np.load(processed_dir / "X_test_final.npy")
    y_test = np.load(processed_dir / "y_test.npy")
    
    model_path = PROJECT_ROOT / "models" / "production_model.pkl"
    if not model_path.exists():
        model_path = PROJECT_ROOT / "models" / "random_forest_model.pkl"
    model = joblib.load(model_path)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    class_names = ["Fiction", "Non Fiction"]
    
    acc = float(accuracy_score(y_test, y_pred))
    prec_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_proba)) if y_proba is not None else None
    
    cm = confusion_matrix(y_test, y_pred).tolist()
    clf_report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    
    # Save Plots
    plots_dir = PROJECT_ROOT / "artifacts" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Confusion matrix plot
    fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax_cm)
    ax_cm.set_title("Amazon Bestseller Model - Confusion Matrix")
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    fig_cm.savefig(plots_dir / "confusion_matrix.png", bbox_inches="tight")
    plt.close(fig_cm)
    
    # ROC Curve
    if y_proba is not None:
        fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
        RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax_roc, name="Random Forest")
        ax_roc.set_title("ROC Curve")
        fig_roc.savefig(plots_dir / "roc_curve.png", bbox_inches="tight")
        plt.close(fig_roc)
        
    # Error Analysis
    raw_test_path = processed_dir / "X_test_raw.csv"
    outputs_dir = PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    if raw_test_path.exists():
        df_test = pd.read_csv(raw_test_path)
    else:
        df_test = pd.DataFrame(X_test, columns=["User Rating", "Reviews_Log", "Price", "Year"])
        
    df_test["Actual_Label"] = [class_names[i] for i in y_test]
    df_test["Predicted_Label"] = [class_names[i] for i in y_pred]
    if y_proba is not None:
        df_test["Prediction_Probability"] = y_proba
    df_test["Is_Misclassified"] = (y_test != y_pred)
    
    error_csv_path = outputs_dir / "error_analysis.csv"
    df_test[df_test["Is_Misclassified"]].to_csv(error_csv_path, index=False)
    
    eval_report = {
        "model_type": type(model).__name__,
        "test_samples": len(y_test),
        "accuracy": acc,
        "precision_weighted": prec_weighted,
        "recall_weighted": rec_weighted,
        "f1_weighted": f1_weighted,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "classification_report": clf_report,
        "misclassifications_count": int(np.sum(y_test != y_pred))
    }
    
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / "evaluation_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=4)
        
    print(f"[SUCCESS] Evaluation complete. Accuracy: {acc:.4f}, F1-Weighted: {f1_weighted:.4f}, ROC-AUC: {roc_auc:.4f}")
    print(f"Evaluation report: {report_file}")
    return eval_report

if __name__ == "__main__":
    evaluate_model()

