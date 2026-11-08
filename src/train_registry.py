import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

from pathlib import Path
import joblib
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

try:
    from src.config import (
        X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH,
        PREPROCESSOR_PATH, DATASET_METADATA_PATH, PRODUCTION_MODEL_PATH,
        MLFLOW_EXPERIMENT_NAME, REGISTERED_MODEL_NAME, MLFLOW_TRACKING_URI,
        RANDOM_STATE
    )
    from src.utils import get_logger
    logger = get_logger("Train_Registry_Lab6")
except ImportError:
    BASE_DIR = Path(__file__).resolve().parent.parent
    PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
    MODELS_DIR = BASE_DIR / "models"
    X_TRAIN_PATH = PROCESSED_DATA_DIR / "X_train_final.npy"
    X_TEST_PATH = PROCESSED_DATA_DIR / "X_test_final.npy"
    Y_TRAIN_PATH = PROCESSED_DATA_DIR / "y_train.npy"
    Y_TEST_PATH = PROCESSED_DATA_DIR / "y_test.npy"
    PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
    DATASET_METADATA_PATH = PROCESSED_DATA_DIR / "dataset_metadata.json"
    PRODUCTION_MODEL_PATH = MODELS_DIR / "production_model.pkl"
    MLFLOW_EXPERIMENT_NAME = "Amazon_Bestseller_Genre_Prediction"
    REGISTERED_MODEL_NAME = "Amazon_Bestseller_Production_Model"
    MLFLOW_TRACKING_URI = (BASE_DIR / "mlruns").as_uri()
    RANDOM_STATE = 42
    import logging
    logger = logging.getLogger("Train_Registry_Lab6")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

def train_and_register_model(version_tag="V1", params=None):
    """Trains the candidate Amazon Bestseller genre model and pushes it to the MLflow Model Registry."""
    logger.info(f"--- Starting MLflow Run with Model Registry Registration ({version_tag}) ---")

    # 1. Set up MLflow tracking
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # 2. Load Processed Data
    try:
        X_train = np.load(str(X_TRAIN_PATH))
        X_test = np.load(str(X_TEST_PATH))
        y_train = np.load(str(Y_TRAIN_PATH))
        y_test = np.load(str(Y_TEST_PATH))
    except FileNotFoundError:
        logger.error("[ERROR] Processed data not found. Please run preprocessing first.")
        return None

    # 3. Model Parameters (Defaulting to Tuned Random Forest from amazonnn.ipynb Cell 93)
    if params is None:
        params = {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 5,
            "min_samples_leaf": 1,
            "random_state": RANDOM_STATE
        }

    run_name = f"Amazon_RandomForest_Registry_{version_tag}"
    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        clean_params = {k: (str(v) if isinstance(v, (list, tuple, dict)) else v) for k, v in params.items()}
        mlflow.log_params(clean_params)
        mlflow.log_param("model_family", "RandomForest")
        mlflow.log_param("version_tag", version_tag)
        mlflow.log_param("task", "Amazon_Bestseller_Genre_Classification")

        # 4. Train Model
        logger.info(f"Training RandomForest candidate model ({run_name})...")
        candidate_model = RandomForestClassifier(**params)
        candidate_model.fit(X_train, y_train)

        # 5. Evaluate on Test Set
        logger.info("Evaluating candidate model on test set...")
        y_pred = candidate_model.predict(X_test)
        y_proba = candidate_model.predict_proba(X_test)[:, 1] if hasattr(candidate_model, "predict_proba") else None

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1_score_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)) if y_proba is not None else 0.0
        }
        mlflow.log_metrics(metrics)

        # 6. Attach Preprocessor Pipeline Artifact (Lineage)
        if os.path.exists(str(PREPROCESSOR_PATH)):
            logger.info("Attaching preprocessing pipeline artifact to model run...")
            mlflow.log_artifact(str(PREPROCESSOR_PATH), artifact_path="preprocessing_pipeline")

        if os.path.exists(str(DATASET_METADATA_PATH)):
            mlflow.log_artifact(str(DATASET_METADATA_PATH), artifact_path="metadata")

        # 7. Log and Register Model in MLflow Registry
        logger.info(f"Registering model under name: '{REGISTERED_MODEL_NAME}'...")
        model_info = mlflow.sklearn.log_model(
            sk_model=candidate_model,
            name="amazon_bestseller_model",
            registered_model_name=REGISTERED_MODEL_NAME,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
        )

        # Also save local production backup
        joblib.dump(candidate_model, str(PRODUCTION_MODEL_PATH))

        logger.info(f"[SUCCESS] Metrics: Test F1={metrics['f1_score_weighted']:.4f} | Accuracy={metrics['accuracy']:.4f} | ROC-AUC={metrics['roc_auc']:.4f}")
        logger.info(f"[SUCCESS] Model successfully registered under '{REGISTERED_MODEL_NAME}' in MLflow Registry!")
        return run.info.run_id, metrics

if __name__ == "__main__":
    train_and_register_model()
