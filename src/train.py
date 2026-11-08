import os
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

try:
    from src.config import (
        X_TRAIN_PATH, Y_TRAIN_PATH, BASELINE_MODEL_PATH,
        RANDOM_FOREST_MODEL_PATH, MODELS_DIR, RANDOM_STATE
    )
    from src.utils import get_logger
    logger = get_logger("Train_Baseline_Lab3")
except ImportError:
    BASE_DIR = Path(__file__).resolve().parent.parent
    PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
    MODELS_DIR = BASE_DIR / "models"
    X_TRAIN_PATH = PROCESSED_DATA_DIR / "X_train_final.npy"
    Y_TRAIN_PATH = PROCESSED_DATA_DIR / "y_train.npy"
    BASELINE_MODEL_PATH = MODELS_DIR / "baseline_model.pkl"
    RANDOM_FOREST_MODEL_PATH = MODELS_DIR / "random_forest_model.pkl"
    RANDOM_STATE = 42
    import logging
    logger = logging.getLogger("Train_Baseline_Lab3")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

def run_training():
    """Trains the baseline Amazon Bestselling Books classifier (Random Forest baseline from notebook)."""
    logger.info("Starting Amazon Baseline Model Training (Lab 3)...")
    os.makedirs(str(MODELS_DIR), exist_ok=True)

    # 1. Load processed training data
    if not os.path.exists(str(X_TRAIN_PATH)) or not os.path.exists(str(Y_TRAIN_PATH)):
        logger.error(f"Processed training data not found at {X_TRAIN_PATH} or {Y_TRAIN_PATH}")
        raise FileNotFoundError("Processed training data not found. Please run preprocessing first.")

    X_train_final = np.load(str(X_TRAIN_PATH))
    y_train = np.load(str(Y_TRAIN_PATH))
    logger.info(f"Loaded processed training data: X_train={X_train_final.shape}, y_train={y_train.shape}")

    # 2. Train baseline Random Forest model from notebook (Cell 87)
    # Notebook parameters: n_estimators=200, random_state=42
    baseline_model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE
    )

    logger.info("Fitting RandomForestClassifier(n_estimators=200, random_state=42)...")
    baseline_model.fit(X_train_final, y_train)

    # 3. Save baseline model artifacts
    joblib.dump(baseline_model, str(BASELINE_MODEL_PATH))
    joblib.dump(baseline_model, str(RANDOM_FOREST_MODEL_PATH))
    logger.info(f"Saved baseline model artifact to '{BASELINE_MODEL_PATH}' and '{RANDOM_FOREST_MODEL_PATH}'")

    logger.info("[SUCCESS] Baseline model training complete!")
    return baseline_model

if __name__ == "__main__":
    run_training()
