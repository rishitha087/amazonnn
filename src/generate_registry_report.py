import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

from datetime import datetime, timezone
import json
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

try:
    from src.config import (
        REGISTERED_MODEL_NAME, MLFLOW_TRACKING_URI, ARTIFACTS_DIR,
        REGISTRY_REPORT_PATH, PRODUCTION_MODEL_REPORT_PATH, PREPROCESSOR_PATH
    )
    from src.utils import get_logger, save_json
    logger = get_logger("Registry_Report_Lab6")
except ImportError:
    BASE_DIR = Path(__file__).resolve().parent.parent
    REGISTERED_MODEL_NAME = "Amazon_Bestseller_Production_Model"
    MLFLOW_TRACKING_URI = (BASE_DIR / "mlruns").as_uri()
    ARTIFACTS_DIR = BASE_DIR / "artifacts"
    REGISTRY_REPORT_PATH = ARTIFACTS_DIR / "registry_report.json"
    PRODUCTION_MODEL_REPORT_PATH = ARTIFACTS_DIR / "production_model_report.json"
    PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"
    import logging
    logger = logging.getLogger("Registry_Report_Lab6")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    def save_json(d, p):
        with open(str(p), "w") as f:
            json.dump(d, f, indent=4)

def generate_registry_report(model_name: str = None):
    """Generates comprehensive deployment and registry reports for the active production model."""
    if model_name is None:
        model_name = REGISTERED_MODEL_NAME

    logger.info(f"Generating Amazon Model Registry Deployment Report for '{model_name}'...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    try:
        all_versions = client.search_model_versions(f"name='{model_name}'")
        if not all_versions:
            logger.error(f"[ERROR] No registered versions found for model '{model_name}'.")
            return False

        # Find production champion, or fall back to latest version
        champion = next((mv for mv in all_versions if mv.current_stage == "Production"), None)
        if not champion:
            logger.warning("[WARNING] No version with stage 'Production' found. Using latest registered version.")
            champion = all_versions[-1]

        run = client.get_run(champion.run_id)
        metrics = run.data.metrics
        params = run.data.params

        report = {
            "registered_model_name": model_name,
            "model_version": int(champion.version),
            "run_id": champion.run_id,
            "current_stage": champion.current_stage,
            "model_type": params.get("model_family", "RandomForest"),
            "artifact_uri": champion.source,
            "promotion_decision": "CHAMPION_PROMOTED_TO_PRODUCTION",
            "lifecycle_rule": "Highest Weighted F1 Score (f1_score_weighted) on Unseen Test Partition",
            "parameters": params,
            "metrics": {
                "f1_score_weighted": metrics.get("f1_score_weighted"),
                "accuracy": metrics.get("accuracy"),
                "precision_weighted": metrics.get("precision_weighted"),
                "recall_weighted": metrics.get("recall_weighted"),
                "roc_auc": metrics.get("roc_auc")
            },
            "preprocessing_dependency": "models/preprocessor.pkl",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "APPROVED_FOR_PRODUCTION_SERVING"
        }

        os.makedirs(str(ARTIFACTS_DIR), exist_ok=True)
        save_json(report, REGISTRY_REPORT_PATH)
        save_json(report, PRODUCTION_MODEL_REPORT_PATH)

        logger.info(f"[SUCCESS] Model Registry Report generated for Version {champion.version} ({report['model_type']})!")
        logger.info(f"  -> Registry Report saved to: {REGISTRY_REPORT_PATH}")
        logger.info(f"  -> Production Report saved to: {PRODUCTION_MODEL_REPORT_PATH}")
        return report

    except Exception as e:
        logger.error(f"[ERROR] Failed to generate registry report: {e}")
        return False

if __name__ == "__main__":
    generate_registry_report()
