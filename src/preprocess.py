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
    logger = get_logger("Preprocess_Lab1")
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
    logger = logging.getLogger("Preprocess_Lab1")
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


def run_preprocessing():
    """Executes Lab 1: Data ingestion, feature engineering, split, leakage-free scaling, and artifact export."""
    logger.info("Starting Amazon Bestselling Books Preprocessing Pipeline (Lab 1)...")

    # Ensure target directories exist
    os.makedirs(str(PROCESSED_DATA_DIR), exist_ok=True)
    os.makedirs(str(MODELS_DIR), exist_ok=True)

    # 1. Load the raw Amazon dataset
    data_path = RAW_DATA_PATH if RAW_DATA_PATH.exists() else ALT_RAW_DATA_PATH
    if not data_path.exists():
        logger.error(f"Amazon dataset not found at {RAW_DATA_PATH} or {ALT_RAW_DATA_PATH}")
        raise FileNotFoundError(f"Amazon dataset not found at {data_path}")

    df = pd.read_csv(str(data_path))
    logger.info(f"Loaded raw dataset from '{data_path.name}' with shape: {df.shape}")

    # 2. Inspect and Clean
    null_counts = df.isnull().sum().to_dict()
    logger.info(f"Missing value counts per column: {null_counts}")
    df = df.dropna(subset=[TARGET_COLUMN])

    dup_count = int(df.duplicated().sum())
    logger.info(f"Exact duplicate rows detected: {dup_count}")

    # Enforce correct data types
    df["User Rating"] = pd.to_numeric(df["User Rating"], errors="coerce")
    df["Reviews"] = pd.to_numeric(df["Reviews"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

    # 3. Input Features and Target Separation
    X = df[RAW_FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    # Encode Target (Fiction -> 0, Non Fiction -> 1)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_mapping = {label: int(idx) for idx, label in enumerate(label_encoder.classes_)}
    logger.info(f"Target '{TARGET_COLUMN}' encoded: {class_mapping}")

    # 4. Train/Test Split (Prevent Data Leakage: Split before scaling)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded
    )
    logger.info(
        f"Train/Test split complete: Train={X_train.shape[0]} samples, Test={X_test.shape[0]} samples (Test Size={TEST_SIZE})"
    )

    # 5. Fit Preprocessing Pipeline strictly on Training Data
    preprocessor = create_preprocessor_pipeline()
    X_train_final = preprocessor.fit_transform(X_train)
    X_test_final = preprocessor.transform(X_test)
    logger.info("Modular Preprocessor fitted strictly on training partition and transformed test partition.")

    # 6. Fit standalone StandardScaler on engineered features for model backward compatibility
    df_train_engineered = X_train.copy()
    df_train_engineered["Reviews_Log"] = np.log1p(df_train_engineered["Reviews"])
    scaler = StandardScaler()
    scaler.fit(df_train_engineered[ENGINEERED_FEATURE_COLUMNS])

    # 7. Save Processed Data Arrays
    np.save(str(X_TRAIN_PATH), X_train_final)
    np.save(str(X_TEST_PATH), X_test_final)
    np.save(str(Y_TRAIN_PATH), y_train.astype(np.int64))
    np.save(str(Y_TEST_PATH), y_test.astype(np.int64))
    logger.info(f"Saved processed NumPy arrays to '{PROCESSED_DATA_DIR}'")

    # 8. Save Preprocessing Artifacts
    joblib.dump(scaler, str(SCALER_PATH))
    joblib.dump(label_encoder, str(ENCODER_PATH))
    joblib.dump(preprocessor, str(PREPROCESSOR_PATH))
    logger.info(f"Saved preprocessing artifacts: scaler -> {SCALER_PATH}, encoder -> {ENCODER_PATH}, preprocessor -> {PREPROCESSOR_PATH}")

    # 9. Save Dataset Metadata
    metadata = {
        "dataset_name": "Amazon Top 50 Bestselling Books (2009-2019)",
        "source_file": str(data_path),
        "total_records": int(len(df)),
        "train_samples": int(X_train_final.shape[0]),
        "test_samples": int(X_test_final.shape[0]),
        "num_features": int(X_train_final.shape[1]),
        "raw_features": RAW_FEATURE_COLUMNS,
        "engineered_features": ENGINEERED_FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "classes": label_encoder.classes_.tolist(),
        "class_mapping": class_mapping,
        "train_class_distribution": {str(k): int(v) for k, v in pd.Series(y_train).value_counts().items()},
        "test_class_distribution": {str(k): int(v) for k, v in pd.Series(y_test).value_counts().items()},
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE
    }
    save_json(metadata, DATASET_METADATA_PATH)
    logger.info(f"Saved dataset metadata to '{DATASET_METADATA_PATH}'")

    logger.info("[SUCCESS] Lab 1 Data Preprocessing completed successfully!")
    return X_train_final, X_test_final, y_train, y_test

if __name__ == "__main__":
    run_preprocessing()
