import os
import sys
import json
import yaml
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

def manage_lifecycle():
    print("=" * 60)
    print("STAGE 8: MLFLOW MODEL LIFECYCLE AUTOMATION & STAGE TRANSITIONS")
    print("=" * 60)
    
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    db_path = PROJECT_ROOT / "mlflow.db"
    tracking_uri = f"sqlite:///{db_path.as_posix()}"
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    
    client = MlflowClient(tracking_uri=tracking_uri)
    model_name = config.get("registry", {}).get("model_name", "Amazon_Bestseller_Production_Model")
    
    try:
        latest_versions = client.get_latest_versions(model_name)
    except Exception as e:
        print(f"[WARN] Could not retrieve latest versions from registry: {e}")
        latest_versions = []
        
    current_prod = None
    candidate = None
    
    for v in latest_versions:
        if v.current_stage == "Production":
            current_prod = v
        elif v.current_stage in ["Staging", "None"]:
            candidate = v
            
    # If no candidate found but versions exist, take the highest version
    if candidate is None and latest_versions:
        candidate = sorted(latest_versions, key=lambda x: int(x.version))[-1]
        
    lifecycle_action = "NO_ACTION"
    if candidate is not None:
        if current_prod is None:
            # Promote candidate to Production directly
            client.transition_model_version_stage(
                name=model_name,
                version=candidate.version,
                stage="Production",
                archive_existing_versions=True
            )
            lifecycle_action = f"Promoted version {candidate.version} to Production (Initial Champion)"
        else:
            # Check metrics
            cand_run = client.get_run(candidate.run_id)
            prod_run = client.get_run(current_prod.run_id)
            
            cand_f1 = float(cand_run.data.metrics.get("f1_score_weighted", 0.0))
            prod_f1 = float(prod_run.data.metrics.get("f1_score_weighted", 0.0))
            
            if cand_f1 >= prod_f1:
                client.transition_model_version_stage(
                    name=model_name,
                    version=candidate.version,
                    stage="Production",
                    archive_existing_versions=True
                )
                lifecycle_action = f"Promoted version {candidate.version} (F1={cand_f1:.4f}) over version {current_prod.version} (F1={prod_f1:.4f})"
            else:
                client.transition_model_version_stage(
                    name=model_name,
                    version=candidate.version,
                    stage="Staging"
                )
                lifecycle_action = f"Kept version {candidate.version} in Staging (F1={cand_f1:.4f} < Prod F1={prod_f1:.4f})"
                
    lifecycle_summary = {
        "model_name": model_name,
        "action": lifecycle_action,
        "active_production_version": candidate.version if candidate and "Promoted" in lifecycle_action else (current_prod.version if current_prod else "N/A"),
        "timestamp": str(Path(__file__).stat().st_mtime)
    }
    
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / "lifecycle_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(lifecycle_summary, f, indent=4)
        
    print(f"[SUCCESS] Model Lifecycle transition complete: {lifecycle_action}")
    print(f"Report saved to: {report_file}")
    return lifecycle_summary

if __name__ == "__main__":
    manage_lifecycle()
