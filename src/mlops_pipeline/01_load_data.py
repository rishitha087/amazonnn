import os
import json
import yaml
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_data():
    print("=" * 60)
    print("STAGE 1: AMAZON BESTSELLERS DATA LOADING & EDA")
    print("=" * 60)
    
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    raw_path = PROJECT_ROOT / config["data"]["raw_path"]
    if not raw_path.exists():
        alt_path = PROJECT_ROOT / config["data"]["alt_raw_path"]
        if alt_path.exists():
            import shutil
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(alt_path, raw_path)
        else:
            raise FileNotFoundError(f"Raw dataset not found at {raw_path}")
            
    df = pd.read_csv(raw_path)
    print(f"[INFO] Loaded Amazon Bestsellers dataset: {raw_path}")
    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}")
    
    # Quick EDA
    missing_counts = df.isnull().sum().to_dict()
    duplicates = int(df.duplicated().sum())
    target_dist = df[config["data"]["target_column"]].value_counts().to_dict()
    
    print(f"[INFO] Missing values per column: {missing_counts}")
    print(f"[INFO] Duplicate rows: {duplicates}")
    print(f"[INFO] Genre distribution: {target_dist}")
    
    report = {
        "dataset_name": "Amazon Top 50 Bestselling Books (2009-2019)",
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "target_column": config["data"]["target_column"],
        "target_distribution": target_dist,
        "missing_values": missing_counts,
        "duplicate_rows": duplicates,
        "raw_features": config["data"]["raw_features"]
    }
    
    artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    report_path = artifacts_dir / "data_load_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
        
    print(f"[SUCCESS] Data load report saved to {report_path}")
    return df

if __name__ == "__main__":
    load_data()

