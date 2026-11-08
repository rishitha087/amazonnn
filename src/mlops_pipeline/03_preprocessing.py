import os
import sys
import json
import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def run_preprocessing():
    print("=" * 60)
    print("STAGE 3: FEATURE ENGINEERING, SCALING & PREPROCESSING")
    print("=" * 60)
    
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    raw_path = PROJECT_ROOT / config["data"]["raw_path"]
    df = pd.read_csv(raw_path)
    
    # Clean and encode
    df = df.copy()
    df["Genre_Encoded"] = df["Genre"].map({"Fiction": 0, "Non Fiction": 1})
    df["Reviews_Log"] = np.log1p(df["Reviews"])
    
    feature_cols = ["User Rating", "Reviews_Log", "Price", "Year"]
    target_col = "Genre_Encoded"
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Train / Test split
    test_size = float(config["split"]["test_size"])
    random_state = int(config["split"]["random_state"])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Fit scaler strictly on train set
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Also save ColumnTransformer for serving
    preprocessor = ColumnTransformer(
        transformers=[
            ("scaler", StandardScaler(), feature_cols)
        ]
    )
    preprocessor.fit(X_train)
    
    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arrays
    np.save(processed_dir / "X_train_final.npy", X_train_scaled)
    np.save(processed_dir / "X_test_final.npy", X_test_scaled)
    np.save(processed_dir / "y_train.npy", y_train.to_numpy(dtype=np.int64))
    np.save(processed_dir / "y_test.npy", y_test.to_numpy(dtype=np.int64))
    
    # Save raw split CSVs for error analysis alignment
    X_train.to_csv(processed_dir / "X_train_raw.csv", index=False)
    X_test.to_csv(processed_dir / "X_test_raw.csv", index=False)
    
    # Save scaler and preprocessor
    joblib.dump(scaler, models_dir / "scaler.pkl")
    joblib.dump(preprocessor, models_dir / "preprocessor.pkl")
    
    metadata = {
        "dataset": "Amazon Bestselling Books (2009-2019)",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features": feature_cols,
        "feature_dim": len(feature_cols),
        "class_mapping": {"Fiction": 0, "Non Fiction": 1},
        "test_size": test_size,
        "random_state": random_state
    }
    
    with open(processed_dir / "dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
        
    artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    with open(artifacts_dir / "preprocessing_summary_report.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"[SUCCESS] Preprocessing completed. Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    run_preprocessing()

