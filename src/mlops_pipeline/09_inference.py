import os
import sys
import json
import time
import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

import mlflow
import mlflow.sklearn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

def run_inference():
    print("=" * 60)
    print("STAGE 9: BATCH INFERENCE & PRODUCTION PREDICTION PIPELINE")
    print("=" * 60)
    
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    db_path = PROJECT_ROOT / "mlflow.db"
    tracking_uri = f"sqlite:///{db_path.as_posix()}"
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    
    model = None
    model_source = "Local Artifact"
    model_name = config.get("registry", {}).get("model_name", "Amazon_Bestseller_Production_Model")
    
    # Try loading from MLflow registry Production stage first
    try:
        model_uri = f"models:/{model_name}/Production"
        model = mlflow.sklearn.load_model(model_uri)
        model_source = f"MLflow Registry ({model_uri})"
    except Exception as e:
        print(f"[INFO] MLflow load fallback: {e}")
        local_model_path = PROJECT_ROOT / "models" / "production_model.pkl"
        if local_model_path.exists():
            model = joblib.load(local_model_path)
        else:
            fallback_rf = PROJECT_ROOT / "models" / "random_forest_model.pkl"
            model = joblib.load(fallback_rf)
            
    print(f"Loaded model from: {model_source}")
    
    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]
    X_test_scaled = np.load(processed_dir / "X_test_final.npy")
    y_test = np.load(processed_dir / "y_test.npy")
    
    start_time = time.time()
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else [0.5] * len(preds)
    inference_time_ms = (time.time() - start_time) * 1000
    
    class_names = ["Fiction", "Non Fiction"]
    pred_labels = [class_names[p] for p in preds]
    actual_labels = [class_names[a] for a in y_test]
    
    outputs_dir = PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions CSV
    df_preds = pd.DataFrame({
        "Actual_Class": actual_labels,
        "Predicted_Class": pred_labels,
        "Probability_NonFiction": probs,
        "Prediction_Correct": (preds == y_test)
    })
    preds_path = outputs_dir / "inference_predictions.csv"
    df_preds.to_csv(preds_path, index=False)
    
    # Log to inference log
    logs_dir = PROJECT_ROOT / "logs" / "inference"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_source": model_source,
        "batch_size": len(preds),
        "latency_ms": round(inference_time_ms, 2),
        "accuracy_on_batch": round(float(np.mean(preds == y_test)), 4),
        "predicted_fiction_count": int(np.sum(preds == 0)),
        "predicted_nonfiction_count": int(np.sum(preds == 1))
    }
    with open(logs_dir / "inference_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")
        
    artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    with open(artifacts_dir / "inference_summary.json", "w", encoding="utf-8") as f:
        json.dump(log_entry, f, indent=4)
        
    print(f"[SUCCESS] Inference complete. Samples: {len(preds)}, Latency: {inference_time_ms:.2f} ms")
    print(f"Predictions saved to: {preds_path}")
    return df_preds

if __name__ == "__main__":
    run_inference()
