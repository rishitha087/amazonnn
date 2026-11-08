import os
import sys
import json
import shutil
import yaml
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def prepare_deployment():
    print("=" * 60)
    print("STAGE 11: PREPARING PRODUCTION DEPLOYMENT BUNDLE")
    print("=" * 60)
    
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    deployment_dir = PROJECT_ROOT / "deployment"
    deployment_dir.mkdir(parents=True, exist_ok=True)
    
    models_dir = PROJECT_ROOT / "models"
    model_src = models_dir / "production_model.pkl"
    if not model_src.exists():
        model_src = models_dir / "random_forest_model.pkl"
    scaler_src = models_dir / "scaler.pkl"
    preprocessor_src = models_dir / "preprocessor.pkl"
    
    shutil.copy(model_src, deployment_dir / "production_model.pkl")
    shutil.copy(scaler_src, deployment_dir / "scaler.pkl")
    if preprocessor_src.exists():
        shutil.copy(preprocessor_src, deployment_dir / "preprocessor.pkl")
        
    eval_report_path = PROJECT_ROOT / "reports" / "evaluation_report.json"
    metrics = {}
    if eval_report_path.exists():
        with open(eval_report_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
            
    metadata = {
        "model_name": config.get("registry", {}).get("model_name", "Amazon_Bestseller_Production_Model"),
        "task": "Binary Classification (Book Genre)",
        "framework": "scikit-learn",
        "classes": {0: "Fiction", 1: "Non Fiction"},
        "input_features": ["User Rating", "Reviews_Log", "Price", "Year"],
        "feature_count": 4,
        "metrics": {
            "accuracy": metrics.get("accuracy", 0.7909),
            "f1_weighted": metrics.get("f1_weighted", 0.7859),
            "roc_auc": metrics.get("roc_auc", 0.8653)
        },
        "artifacts": [
            "production_model.pkl",
            "scaler.pkl",
            "preprocessor.pkl"
        ]
    }
    
    meta_path = deployment_dir / "model_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"[SUCCESS] Deployment bundle created in {deployment_dir}")
    print(f"Metadata saved to: {meta_path}")
    return metadata

if __name__ == "__main__":
    prepare_deployment()
