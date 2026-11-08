import os
import sys
import json
import yaml
import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import mlflow
import mlflow.sklearn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

def train_and_register():
    print("=" * 60)
    print("STAGE 5: MODEL TRAINING & MLFLOW REGISTRY")
    print("=" * 60)
    
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    db_path = PROJECT_ROOT / "mlflow.db"
    tracking_uri = f"sqlite:///{db_path.as_posix()}"
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    
    experiment_name = config["mlflow"]["experiment_name"]
    mlflow.set_experiment(experiment_name)
    
    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]
    X_train = np.load(processed_dir / "X_train_final.npy")
    X_test = np.load(processed_dir / "X_test_final.npy")
    y_train = np.load(processed_dir / "y_train.npy")
    y_test = np.load(processed_dir / "y_test.npy")
    
    model_cfg = config.get("model", {})
    rf_cfg = model_cfg.get("random_forest", model_cfg)
    n_estimators = int(rf_cfg.get("n_estimators", 100))
    max_depth = int(rf_cfg.get("max_depth", 5))
    random_state = int(rf_cfg.get("random_state", 42))
    
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_proba))
    
    metrics = {
        "accuracy": acc,
        "precision_weighted": prec,
        "recall_weighted": rec,
        "f1_score_weighted": f1,
        "roc_auc": roc_auc
    }
    
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "random_forest_model.pkl"
    joblib.dump(clf, model_path)
    # Also save as production_model.pkl candidate
    joblib.dump(clf, models_dir / "production_model.pkl")
    
    reg_name = config.get("registry", {}).get("model_name", "Amazon_Bestseller_Production_Model")
    with mlflow.start_run(run_name="RandomForest_Pipeline_Registry") as run:
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_metrics(metrics)
        
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            pip_requirements=["scikit-learn", "cloudpickle", "numpy"],
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            registered_model_name=reg_name
        )
        run_id = run.info.run_id
        
    artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "model_name": reg_name,
        "run_id": run_id,
        "metrics": metrics,
        "hyperparameters": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state
        },
        "model_path": str(model_path)
    }
    with open(artifacts_dir / "training_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
        
    print(f"[SUCCESS] Model trained & registered. Run ID: {run_id}")
    print(f"Metrics: Accuracy={acc:.4f}, F1-Weighted={f1:.4f}, ROC-AUC={roc_auc:.4f}")
    return report

if __name__ == "__main__":
    train_and_register()
