import json
import os
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

try:
    from src.config import (
        X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH,
        ARTIFACTS_DIR, REPRODUCIBILITY_REPORT_PATH, RANDOM_STATE
    )
    from src.utils import get_logger, save_json
    logger = get_logger("Reproducibility_Validation_Lab4")
except ImportError:
    BASE_DIR = Path(__file__).resolve().parent.parent
    PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
    ARTIFACTS_DIR = BASE_DIR / "artifacts"
    X_TRAIN_PATH = PROCESSED_DATA_DIR / "X_train_final.npy"
    X_TEST_PATH = PROCESSED_DATA_DIR / "X_test_final.npy"
    Y_TRAIN_PATH = PROCESSED_DATA_DIR / "y_train.npy"
    Y_TEST_PATH = PROCESSED_DATA_DIR / "y_test.npy"
    REPRODUCIBILITY_REPORT_PATH = ARTIFACTS_DIR / "reproducibility_report.json"
    RANDOM_STATE = 42
    import logging
    logger = logging.getLogger("Reproducibility_Validation_Lab4")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    def save_json(d, p):
        with open(str(p), "w") as f:
            json.dump(d, f, indent=4)

def run_deterministic_test():
    """Runs dual deterministic execution test to verify reproducibility of training and evaluation."""
    logger.info("Running Amazon Pipeline Reproducibility Validation (Lab 4)...")

    # Load preprocessed arrays
    X_train = np.load(str(X_TRAIN_PATH))
    X_test = np.load(str(X_TEST_PATH))
    y_train = np.load(str(Y_TRAIN_PATH))
    y_test = np.load(str(Y_TEST_PATH))

    # Fixed parameters for strict deterministic execution
    params = {
        "n_estimators": 200,
        "max_depth": None,
        "random_state": RANDOM_STATE
    }

    # Execution 1
    model_1 = RandomForestClassifier(**params)
    model_1.fit(X_train, y_train)
    preds_1 = model_1.predict(X_test)
    f1_1 = float(f1_score(y_test, preds_1, average="weighted"))
    acc_1 = float(accuracy_score(y_test, preds_1))

    # Execution 2 (Simulating pipeline re-run)
    model_2 = RandomForestClassifier(**params)
    model_2.fit(X_train, y_train)
    preds_2 = model_2.predict(X_test)
    f1_2 = float(f1_score(y_test, preds_2, average="weighted"))
    acc_2 = float(accuracy_score(y_test, preds_2))

    # Evaluate consistency
    preds_identical = bool(np.array_equal(preds_1, preds_2))
    f1_identical = bool(abs(f1_1 - f1_2) < 1e-9)
    acc_identical = bool(abs(acc_1 - acc_2) < 1e-9)
    is_strictly_reproducible = preds_identical and f1_identical and acc_identical

    report = {
        "pipeline": "Amazon Bestselling Books MLOps",
        "test_name": "Pipeline Determinism and Reproducibility Validation",
        "fixed_seed": RANDOM_STATE,
        "model_architecture": "RandomForestClassifier",
        "parameters": params,
        "execution_1": {
            "weighted_f1": f1_1,
            "accuracy": acc_1
        },
        "execution_2": {
            "weighted_f1": f1_2,
            "accuracy": acc_2
        },
        "predictions_identical": preds_identical,
        "metrics_identical": f1_identical,
        "is_strictly_reproducible": is_strictly_reproducible,
        "status": "PASSED" if is_strictly_reproducible else "FAILED"
    }

    os.makedirs(str(ARTIFACTS_DIR), exist_ok=True)
    save_json(report, REPRODUCIBILITY_REPORT_PATH)

    logger.info(f"Execution 1 Weighted F1: {f1_1:.6f} | Accuracy: {acc_1:.6f}")
    logger.info(f"Execution 2 Weighted F1: {f1_2:.6f} | Accuracy: {acc_2:.6f}")
    logger.info(f"Predictions Identical: {preds_identical}")

    if is_strictly_reproducible:
        logger.info(f"[SUCCESS] Amazon pipeline is 100% reproducible! Report saved to '{REPRODUCIBILITY_REPORT_PATH}'.")
    else:
        logger.error("[FAILED] Pipeline execution produced non-deterministic outputs.")

    return is_strictly_reproducible

if __name__ == "__main__":
    run_deterministic_test()
