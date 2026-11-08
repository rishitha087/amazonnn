import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

try:
    from src.config import (
        RAW_DATA_PATH, ALT_RAW_DATA_PATH, PROCESSED_DATA_DIR, MODELS_DIR,
        X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH,
        SCALER_PATH, ENCODER_PATH, PREPROCESSOR_PATH, DATASET_METADATA_PATH,
        RANDOM_STATE, TEST_SIZE, TARGET_COLUMN, RAW_FEATURE_COLUMNS,
        ENGINEERED_FEATURE_COLUMNS, CLASS_NAMES
    )
    from src.utils import get_logger, save_json
    logger = get_logger("Preprocess_Pipeline_Lab5")
except ImportError:
    BASE_DIR = Path(__file__).resolve().parent.parent
    RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "amazon_bestsellers.csv"
    ALT_RAW_DATA_PATH = BASE_DIR / "bestsellers with categories.csv"
    PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
    MODELS_DIR = BASE_DIR / "models"
    X_TRAIN_PATH = PROCESSED_DATA_DIR / "X_train_final.npy"
    X_TEST_PATH = PROCESSED_DATA_DIR / "X_test_final.npy"
    Y_TRAIN_PATH = PROCESSED_DATA_DIR / "y_train.npy"
    Y_TEST_PATH = PROCESSED_DATA_DIR / "y_test.npy"
    SCALER_PATH = MODELS_DIR / "scaler.pkl"
    ENCODER_PATH = MODELS_DIR / "encoder.pkl"
    PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
    DATASET_METADATA_PATH = PROCESSED_DATA_DIR / "dataset_metadata.json"
    RANDOM_STATE = 42
    TEST_SIZE = 0.20
    TARGET_COLUMN = "Genre"
    RAW_FEATURE_COLUMNS = ["User Rating", "Reviews", "Price", "Year"]
    ENGINEERED_FEATURE_COLUMNS = ["User Rating", "Reviews_Log", "Price", "Year"]
    CLASS_NAMES = ["Fiction", "Non Fiction"]
    import logging
    logger = logging.getLogger("Preprocess_Pipeline_Lab5")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    def save_json(d, p):
        with open(str(p), "w") as f:
            json.dump(d, f, indent=4)


def create_preprocessor_pipeline():
    """Builds a pure Scikit-Learn ColumnTransformer pipeline matching the notebook transformation."""
    return ColumnTransformer(
        transformers=[
            ("user_rating", StandardScaler(), ["User Rating"]),
            ("reviews_pipe", Pipeline([
                ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                ("scale", StandardScaler())
            ]), ["Reviews"]),
            ("price", StandardScaler(), ["Price"]),
            ("year", StandardScaler(), ["Year"])
        ]
    )


def run_production_preprocessing():
    """Executes the production-grade modular preprocessing pipeline for Amazon Bestsellers (Lab 5)."""
    logger.info("Starting Amazon Production Preprocessing Pipeline (Lab 5)...")

    # 1. Load the raw dataset
    data_path = RAW_DATA_PATH if RAW_DATA_PATH.exists() else ALT_RAW_DATA_PATH
    if not os.path.exists(str(data_path)):
        logger.error(f"Raw data file not found at: {data_path}")
        raise FileNotFoundError(f"Raw data not found at {data_path}")

    df = pd.read_csv(str(data_path))
    logger.info(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.")

    # 2. Extract X and y
    X_raw = df[RAW_FEATURE_COLUMNS].copy()
    y_raw = df[TARGET_COLUMN].copy()

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)
    class_mapping = {label: int(idx) for idx, label in enumerate(label_encoder.classes_)}

    # 3. Train-Test Split (strictly before fitting preprocessor to eliminate leakage)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded
    )
    logger.info(f"Split data into train ({len(X_train_raw)}) and test ({len(X_test_raw)}) partitions.")

    # 4. Build and Fit Production Preprocessor
    preprocessor = create_preprocessor_pipeline()
    logger.info("Fitting Scikit-Learn ColumnTransformer strictly on training partition...")
    X_train_final = preprocessor.fit_transform(X_train_raw)

    logger.info("Transforming test partition with fitted preprocessor...")
    X_test_final = preprocessor.transform(X_test_raw)

    # 5. Fit standalone StandardScaler for backward compatibility
    df_train_engineered = X_train_raw.copy()
    df_train_engineered["Reviews_Log"] = np.log1p(df_train_engineered["Reviews"])
    scaler = StandardScaler()
    scaler.fit(df_train_engineered[ENGINEERED_FEATURE_COLUMNS])

    # 6. Save Artifacts and Data Arrays
    os.makedirs(str(PROCESSED_DATA_DIR), exist_ok=True)
    os.makedirs(str(MODELS_DIR), exist_ok=True)

    np.save(str(X_TRAIN_PATH), X_train_final)
    np.save(str(X_TEST_PATH), X_test_final)
    np.save(str(Y_TRAIN_PATH), y_train.astype(np.int64))
    np.save(str(Y_TEST_PATH), y_test.astype(np.int64))

    joblib.dump(preprocessor, str(PREPROCESSOR_PATH))
    joblib.dump(scaler, str(SCALER_PATH))
    joblib.dump(label_encoder, str(ENCODER_PATH))

    # 7. Save Metadata
    metadata = {
        "dataset_name": "Amazon Top 50 Bestselling Books (2009-2019)",
        "train_shape": list(X_train_final.shape),
        "test_shape": list(X_test_final.shape),
        "raw_features": RAW_FEATURE_COLUMNS,
        "features": ENGINEERED_FEATURE_COLUMNS,
        "classes": label_encoder.classes_.tolist(),
        "class_mapping": class_mapping,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE
    }
    save_json(metadata, DATASET_METADATA_PATH)

    logger.info(f"[SUCCESS] Modular production preprocessor fitted and saved to '{PREPROCESSOR_PATH}'.")
    return preprocessor, X_train_final, X_test_final, y_train, y_test

if __name__ == "__main__":
    run_production_preprocessing()
