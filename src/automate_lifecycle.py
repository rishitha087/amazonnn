import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

try:
    from src.config import REGISTERED_MODEL_NAME, MLFLOW_TRACKING_URI
    from src.utils import get_logger
    logger = get_logger("Automate_Lifecycle_Lab6")
except ImportError:
    BASE_DIR = Path(__file__).resolve().parent.parent
    REGISTERED_MODEL_NAME = "Amazon_Bestseller_Production_Model"
    MLFLOW_TRACKING_URI = (BASE_DIR / "mlruns").as_uri()
    import logging
    logger = logging.getLogger("Automate_Lifecycle_Lab6")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

def automate_champion_challenger(model_name: str = None, metric_to_optimize: str = "f1_score_weighted"):
    """Automated Champion/Challenger evaluation and lifecycle management for Amazon Bestseller models.
    
    Promotion Rule:
    - Candidate models in 'None' or 'Staging' are evaluated against the current 'Production' champion.
    - The metric used for comparison is Weighted F1-Score ('f1_score_weighted').
    - If no Champion exists, the best candidate is automatically promoted.
    - If Champion exists, Challenger must achieve >= Champion F1-Score to earn promotion.
    - Defeated Champions are safely transitioned to 'Archived'.
    """
    if model_name is None:
        model_name = REGISTERED_MODEL_NAME

    logger.info(f"Starting Automated Model Lifecycle Manager for: '{model_name}'")
    logger.info(f"Optimization Metric: '{metric_to_optimize}'")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    try:
        all_versions = client.search_model_versions(f"name='{model_name}'")
        if not all_versions:
            logger.warning(f"No model versions found in registry for '{model_name}'.")
            return False

        logger.info(f"Found {len(all_versions)} total registered version(s) in registry.")

        # 1. Discover untracked versions (Stage: None) and push to Staging
        new_versions = [mv for mv in all_versions if mv.current_stage == "None"]
        for mv in new_versions:
            logger.info(f"Moving Version {mv.version} from None -> Staging...")
            try:
                client.transition_model_version_stage(
                    name=model_name, version=mv.version, stage="Staging", archive_existing_versions=False
                )
            except Exception as e:
                logger.warning(f"Stage transition notice: {e}")
            try:
                client.set_registered_model_alias(name=model_name, alias="staging", version=mv.version)
            except Exception:
                pass

        # Refresh versions after stage updates
        all_versions = client.search_model_versions(f"name='{model_name}'")
        staging_models = [mv for mv in all_versions if mv.current_stage == "Staging"]

        if not staging_models:
            logger.info("No candidates in Staging stage to evaluate. Checking if Production model exists...")
            prod = [mv for mv in all_versions if mv.current_stage == "Production"]
            if prod:
                logger.info(f"Active Production Champion is Version {prod[0].version}.")
                return True
            else:
                # If all are None or Archived, take latest
                latest = all_versions[-1]
                logger.info(f"Promoting latest Version {latest.version} to Production...")
                client.transition_model_version_stage(name=model_name, version=latest.version, stage="Production")
                client.set_registered_model_alias(name=model_name, alias="champion", version=latest.version)
                return True

        # 2. Evaluate Staging Candidates (Find Best Challenger)
        logger.info("Evaluating Staging candidate models...")
        best_challenger = None
        best_challenger_score = -1.0

        for mv in staging_models:
            run = client.get_run(mv.run_id)
            score = run.data.metrics.get(metric_to_optimize)
            if score is None:
                # fallback to generic f1_score if f1_score_weighted is named differently
                score = run.data.metrics.get("f1_score", 0.0)

            logger.info(f"  -> Candidate Version {mv.version} (Run: {mv.run_id[:8]}) | {metric_to_optimize} = {score:.4f}")
            if score > best_challenger_score:
                best_challenger_score = score
                best_challenger = mv

        logger.info(f"Top Challenger: Version {best_challenger.version} with {metric_to_optimize} = {best_challenger_score:.4f}")

        # 3. Locate Active Production Champion
        production_models = [mv for mv in all_versions if mv.current_stage == "Production"]
        current_champion = production_models[0] if production_models else None
        promote_challenger = False

        if not current_champion:
            logger.info("No active model in Production stage. Challenger promoted by default!")
            promote_challenger = True
        else:
            champ_run = client.get_run(current_champion.run_id)
            champ_score = champ_run.data.metrics.get(metric_to_optimize) or champ_run.data.metrics.get("f1_score", 0.0)
            logger.info(f"Current Production Champion: Version {current_champion.version} | {metric_to_optimize} = {champ_score:.4f}")

            # 4. Compare Challenger vs Champion
            if best_challenger_score >= champ_score:
                logger.info(f"[SUCCESS] Challenger (v{best_challenger.version}, {best_challenger_score:.4f}) >= Champion (v{current_champion.version}, {champ_score:.4f}). Promotion approved!")
                promote_challenger = True
            else:
                logger.info(f"[DECISION] Challenger ({best_challenger_score:.4f}) did not exceed Champion ({champ_score:.4f}). Champion retained.")

        # 5. Execute Promotions & Archivals
        if promote_challenger:
            logger.info(f"Promoting Version {best_challenger.version} to 'Production' stage...")
            client.transition_model_version_stage(
                name=model_name, version=best_challenger.version, stage="Production", archive_existing_versions=False
            )
            try:
                client.set_registered_model_alias(name=model_name, alias="champion", version=best_challenger.version)
            except Exception:
                pass

            if current_champion and current_champion.version != best_challenger.version:
                logger.info(f"Archiving defeated Champion (Version {current_champion.version})...")
                client.transition_model_version_stage(
                    name=model_name, version=current_champion.version, stage="Archived", archive_existing_versions=False
                )

        # 6. Clean up remaining staging models
        final_versions = client.search_model_versions(f"name='{model_name}'")
        remaining_staging = [mv for mv in final_versions if mv.current_stage == "Staging"]
        for mv in remaining_staging:
            logger.info(f"Archiving unpromoted staging candidate: Version {mv.version}...")
            client.transition_model_version_stage(
                name=model_name, version=mv.version, stage="Archived", archive_existing_versions=False
            )

        logger.info("[SUCCESS] Automated Model Lifecycle execution completed successfully!")
        return True

    except Exception as e:
        logger.error(f"[ERROR] Model lifecycle automation failed: {e}")
        return False

if __name__ == "__main__":
    automate_champion_challenger()
