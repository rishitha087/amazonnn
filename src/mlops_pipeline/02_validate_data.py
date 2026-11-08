import os
import sys
import json
import yaml
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def validate_data():
    print("=" * 60)
    print("STAGE 2: RAW DATA SCHEMA & QUALITY VALIDATION")
    print("=" * 60)
    
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    raw_path = PROJECT_ROOT / config["data"]["raw_path"]
    if not raw_path.exists():
        print(f"[ERROR] Dataset missing at {raw_path}")
        sys.exit(1)
        
    df = pd.read_csv(raw_path)
    required_columns = ["Name", "Author", "User Rating", "Reviews", "Price", "Year", "Genre"]
    
    errors = []
    # 1. Missing columns
    for col in required_columns:
        if col not in df.columns:
            errors.append(f"MISSING_COLUMN: Required column '{col}' is missing.")
            
    if errors:
        for err in errors:
            print(f"[ERROR] {err}")
        sys.exit(1)
        
    # 2. Data type validation
    if not pd.api.types.is_numeric_dtype(df["User Rating"]):
        errors.append("INVALID_TYPE: 'User Rating' must be numeric.")
    if not pd.api.types.is_numeric_dtype(df["Reviews"]):
        errors.append("INVALID_TYPE: 'Reviews' must be numeric.")
    if not pd.api.types.is_numeric_dtype(df["Price"]):
        errors.append("INVALID_TYPE: 'Price' must be numeric.")
    if not pd.api.types.is_numeric_dtype(df["Year"]):
        errors.append("INVALID_TYPE: 'Year' must be numeric.")
        
    # 3. Value range validation
    if (df["User Rating"] < 0).any() or (df["User Rating"] > 5.0).any():
        errors.append("VALUE_OUT_OF_RANGE: 'User Rating' must be between 0.0 and 5.0.")
    if (df["Price"] < 0).any():
        errors.append("VALUE_OUT_OF_RANGE: 'Price' must be non-negative.")
    if (df["Reviews"] < 0).any():
        errors.append("VALUE_OUT_OF_RANGE: 'Reviews' must be non-negative.")
        
    # 4. Valid target classes
    valid_genres = {"Fiction", "Non Fiction"}
    actual_genres = set(df["Genre"].dropna().unique())
    invalid_genres = actual_genres - valid_genres
    if invalid_genres:
        errors.append(f"INVALID_CATEGORICAL: Unexpected Genre categories: {invalid_genres}")
        
    if errors:
        for err in errors:
            print(f"[ERROR] {err}")
        sys.exit(1)
        
    print(f"[SUCCESS] Schema validation passed! All {len(required_columns)} columns valid across {len(df)} records.")
    return True

if __name__ == "__main__":
    validate_data()

