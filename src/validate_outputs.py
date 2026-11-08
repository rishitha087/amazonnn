import json
import os
from pathlib import Path
import numpy as np

try:
    from src.config import (
        X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH,
        SCALER_PATH, ENCODER_PATH, PREPROCESSOR_PATH,
        ARTIFACTS_DIR, PREPROCESSING_SUMMARY_PATH
    )
    from src.utils import get_logger, save_json
    logger = get_logger("Validate_Outputs_Lab5")
except ImportError:
    BASE_DIR = Path(__file__).resolve().parent.parent
    PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
    MODELS_DIR = BASE_DIR / "models"
    ARTIFACTS_DIR = BASE_DIR / "artifacts"
    X_TRAIN_PATH = PROCESSED_DATA_DIR / "X_train_final.npy"
    X_TEST_PATH = PROCESSED_DATA_DIR / "X_test_final.npy"
    Y_TRAIN_PATH = PROCESSED_DATA_DIR / "y_train.npy"
    Y_TEST_PATH = PROCESSED_DATA_DIR / "y_test.npy"
    SCALER_PATH = MODELS_DIR / "scaler.pkl"
    ENCODER_PATH = MODELS_DIR / "encoder.pkl"
    PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
    PREPROCESSING_SUMMARY_PATH = ARTIFACTS_DIR / "preprocessing_summary_report.json"
    import logging
    logger = logging.getLogger("Validate_Outputs_Lab5")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    def save_json(d, p):
        with open(str(p), "w") as f:
            json.dump(d, f, indent=4)

def validate_preprocessing_outputs():
    """Validates data pipeline outputs, checks dimensionality, absence of NaNs, and artifact integrity."""
    logger.info("Validating Amazon Preprocessing Pipeline Outputs (Lab 5)...")

    errors = []

    # 1. Check array file existence
    for path, name in [
        (X_TRAIN_PATH, "X_train_final"),
        (X_TEST_PATH, "X_test_final"),
        (Y_TRAIN_PATH, "y_train"),
        (Y_TEST_PATH, "y_test"),
    ]:
        if not os.path.exists(str(path)):
            errors.append(f"Missing processed array file: {name} at {path}")

    if errors:
        for err in errors:
            logger.error(f"[ERROR] {err}")
        return False

    # 2. Load arrays
    X_train = np.load(str(X_TRAIN_PATH))
    X_test = np.load(str(X_TEST_PATH))
    y_train = np.load(str(Y_TRAIN_PATH))
    y_test = np.load(str(Y_TEST_PATH))

    # 3. Check for NaNs and Infs
    nan_train = int(np.isnan(X_train).sum())
    nan_test = int(np.isnan(X_test).sum())
    inf_train = int(np.isinf(X_train).sum())
    inf_test = int(np.isinf(X_test).sum())

    if nan_train > 0:
        errors.append(f"Found {nan_train} NaN values in X_train.")
    if nan_test > 0:
        errors.append(f"Found {nan_test} NaN values in X_test.")
    if inf_train > 0:
        errors.append(f"Found {inf_train} Inf values in X_train.")
    if inf_test > 0:
        errors.append(f"Found {inf_test} Inf values in X_test.")

    # 4. Dimensionality checks
    if X_train.shape[1] != X_test.shape[1]:
        errors.append(f"Column count mismatch: X_train={X_train.shape[1]}, X_test={X_test.shape[1]}")
    if X_train.shape[0] != y_train.shape[0]:
        errors.append(f"Row count mismatch between X_train ({X_train.shape[0]}) and y_train ({y_train.shape[0]})")
    if X_test.shape[0] != y_test.shape[0]:
        errors.append(f"Row count mismatch between X_test ({X_test.shape[0]}) and y_test ({y_test.shape[0]})")
    if X_train.shape[1] != 4:
        errors.append(f"Expected 4 engineered features, got {X_train.shape[1]}")

    # 5. Check model artifacts
    artifacts_status = {}
    for art_path, art_name in [
        (PREPROCESSOR_PATH, "preprocessor.pkl"),
        (SCALER_PATH, "scaler.pkl"),
        (ENCODER_PATH, "encoder.pkl"),
    ]:
        exists = os.path.exists(str(art_path))
        artifacts_status[art_name] = "EXISTS" if exists else "MISSING"
        if not exists:
            errors.append(f"Required artifact {art_name} not found at {art_path}")

    # 6. Generate Preprocessing Summary Report
    report = {
        "pipeline": "Amazon Bestsellers Production Preprocessing",
        "validation_status": "PASSED" if not errors else "FAILED",
        "matrix_dimensions": {
            "X_train_shape": list(X_train.shape),
            "X_test_shape": list(X_test.shape),
            "y_train_shape": list(y_train.shape),
            "y_test_shape": list(y_test.shape)
        },
        "data_quality": {
            "nan_count_X_train": nan_train,
            "nan_count_X_test": nan_test,
            "inf_count_X_train": inf_train,
            "inf_count_X_test": inf_test
        },
        "artifacts_status": artifacts_status,
        "errors": errors
    }

    os.makedirs(str(ARTIFACTS_DIR), exist_ok=True)
    save_json(report, PREPROCESSING_SUMMARY_PATH)

    if errors:
        logger.error(f"[FAILED] Output Validation FAILED with {len(errors)} errors:")
        for err in errors:
            logger.error(f"  - {err}")
        return False

    logger.info("[SUCCESS] Output Validation PASSED! Arrays are clean, scaled, and dimensionally verified.")
    logger.info(f"Summary report written to: {PREPROCESSING_SUMMARY_PATH}")
    return True

if __name__ == "__main__":
    validate_preprocessing_outputs()
