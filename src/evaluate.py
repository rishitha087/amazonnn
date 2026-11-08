import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix
)

try:
    from src.config import (
        X_TEST_PATH, Y_TEST_PATH, BASELINE_MODEL_PATH, ENCODER_PATH,
        RAW_DATA_PATH, ALT_RAW_DATA_PATH, OUTPUTS_DIR, METRICS_DIR,
        FALSE_NEGATIVES_PATH, FALSE_POSITIVES_PATH, BASELINE_METRICS_PATH,
        RANDOM_STATE, TEST_SIZE, TARGET_COLUMN
    )
    from src.utils import get_logger, save_json
    logger = get_logger("Evaluate_Baseline_Lab3")
except ImportError:
    BASE_DIR = Path(__file__).resolve().parent.parent
    PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
    MODELS_DIR = BASE_DIR / "models"
    OUTPUTS_DIR = BASE_DIR / "outputs"
    METRICS_DIR = OUTPUTS_DIR / "metrics"
    X_TEST_PATH = PROCESSED_DATA_DIR / "X_test_final.npy"
    Y_TEST_PATH = PROCESSED_DATA_DIR / "y_test.npy"
    BASELINE_MODEL_PATH = MODELS_DIR / "baseline_model.pkl"
    ENCODER_PATH = MODELS_DIR / "encoder.pkl"
    RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "amazon_bestsellers.csv"
    ALT_RAW_DATA_PATH = BASE_DIR / "bestsellers with categories.csv"
    FALSE_NEGATIVES_PATH = OUTPUTS_DIR / "false_negatives.csv"
    FALSE_POSITIVES_PATH = OUTPUTS_DIR / "false_positives.csv"
    BASELINE_METRICS_PATH = METRICS_DIR / "baseline_metrics.json"
    RANDOM_STATE = 42
    TEST_SIZE = 0.20
    TARGET_COLUMN = "Genre"
    import logging
    logger = logging.getLogger("Evaluate_Baseline_Lab3")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    def save_json(d, p):
        with open(str(p), "w") as f:
            json.dump(d, f, indent=4)

def run_evaluation():
    """Evaluates the baseline Amazon Bestseller genre model, logs metrics, and outputs error diagnostics."""
    logger.info("Starting Amazon Baseline Model Evaluation (Lab 3)...")
    os.makedirs(str(OUTPUTS_DIR), exist_ok=True)
    os.makedirs(str(METRICS_DIR), exist_ok=True)

    # 1. Load test data and trained model
    if not os.path.exists(str(X_TEST_PATH)) or not os.path.exists(str(Y_TEST_PATH)):
        logger.error(f"Processed test data missing at {X_TEST_PATH} or {Y_TEST_PATH}")
        raise FileNotFoundError("Processed test data not found.")

    if not os.path.exists(str(BASELINE_MODEL_PATH)):
        logger.error(f"Trained baseline model not found at {BASELINE_MODEL_PATH}")
        raise FileNotFoundError(f"Model not found at {BASELINE_MODEL_PATH}")

    X_test_final = np.load(str(X_TEST_PATH))
    y_test = np.load(str(Y_TEST_PATH))
    model = joblib.load(str(BASELINE_MODEL_PATH))
    encoder = joblib.load(str(ENCODER_PATH)) if os.path.exists(str(ENCODER_PATH)) else None

    # 2. Predict classes and probabilities
    y_pred = model.predict(X_test_final)
    y_proba = model.predict_proba(X_test_final)[:, 1] if hasattr(model, "predict_proba") else None

    # 3. Calculate evaluation metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_proba)) if y_proba is not None else 0.0

    logger.info("--- Amazon Bestseller Baseline Evaluation Report ---")
    logger.info(f"Accuracy          : {acc:.4f}")
    logger.info(f"Precision (weight): {prec_weighted:.4f}")
    logger.info(f"Recall (weight)   : {rec_weighted:.4f}")
    logger.info(f"F1-Score (weight) : {f1_weighted:.4f}")
    logger.info(f"ROC-AUC           : {roc_auc:.4f}")
    logger.info("-----------------------------------------------------")

    # Save metrics JSON
    metrics = {
        "model": "RandomForest_Baseline",
        "dataset": "Amazon Bestselling Books",
        "target": TARGET_COLUMN,
        "test_samples": int(len(y_test)),
        "accuracy": acc,
        "precision_weighted": prec_weighted,
        "recall_weighted": rec_weighted,
        "f1_weighted": f1_weighted,
        "roc_auc": roc_auc
    }
    save_json(metrics, BASELINE_METRICS_PATH)
    logger.info(f"Saved baseline metrics summary to '{BASELINE_METRICS_PATH}'")

    # 4. Error Analysis Export (Misclassified Amazon Books)
    data_path = RAW_DATA_PATH if RAW_DATA_PATH.exists() else ALT_RAW_DATA_PATH
    if data_path.exists():
        raw_df = pd.read_csv(str(data_path))
        _, test_raw_df = train_test_split(
            raw_df,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=raw_df[TARGET_COLUMN]
        )
        
        test_errors = test_raw_df.copy()
        test_errors["Actual_Genre"] = [encoder.classes_[i] if encoder else i for i in y_test]
        test_errors["Predicted_Genre"] = [encoder.classes_[i] if encoder else i for i in y_pred]
        if y_proba is not None:
            test_errors["Predicted_Prob_NonFiction"] = np.round(y_proba, 4)

        # Misclassification breakdown:
        # False Negatives (Actual: Non Fiction (1), Predicted: Fiction (0))
        false_negatives = test_errors[(y_test == 1) & (y_pred == 0)]
        # False Positives (Actual: Fiction (0), Predicted: Non Fiction (1))
        false_positives = test_errors[(y_test == 0) & (y_pred == 1)]

        false_negatives.to_csv(str(FALSE_NEGATIVES_PATH), index=False)
        false_positives.to_csv(str(FALSE_POSITIVES_PATH), index=False)
        logger.info(f"Exported error analysis: {len(false_negatives)} false negatives, {len(false_positives)} false positives.")
        logger.info(f"Saved error reports: {FALSE_NEGATIVES_PATH} & {FALSE_POSITIVES_PATH}")

    logger.info("[SUCCESS] Baseline model evaluation complete!")
    return metrics

if __name__ == "__main__":
    run_evaluation()
