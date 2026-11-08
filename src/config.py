import os
from pathlib import Path

# Set MLflow environment configurations for local Windows environment
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

# Base Paths (Dynamically resolved so scripts work anywhere)
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_DATA_PATH = RAW_DATA_DIR / "amazon_bestsellers.csv"
ALT_RAW_DATA_PATH = BASE_DIR / "bestsellers with categories.csv"

PROCESSED_DATA_DIR = DATA_DIR / "processed"
X_TRAIN_PATH = PROCESSED_DATA_DIR / "X_train_final.npy"
X_TEST_PATH = PROCESSED_DATA_DIR / "X_test_final.npy"
Y_TRAIN_PATH = PROCESSED_DATA_DIR / "y_train.npy"
Y_TEST_PATH = PROCESSED_DATA_DIR / "y_test.npy"
DATASET_METADATA_PATH = PROCESSED_DATA_DIR / "dataset_metadata.json"

# Models Paths
MODELS_DIR = BASE_DIR / "models"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
ENCODER_PATH = MODELS_DIR / "encoder.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
BASELINE_MODEL_PATH = MODELS_DIR / "baseline_model.pkl"
RANDOM_FOREST_MODEL_PATH = MODELS_DIR / "random_forest_model.pkl"
PRODUCTION_MODEL_PATH = MODELS_DIR / "production_model.pkl"
NOTEBOOK_SAVED_MODEL_PATH = BASE_DIR / "bestseller_genre_model.pkl"

# Outputs Paths
OUTPUTS_DIR = BASE_DIR / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
REPORTS_DIR = OUTPUTS_DIR / "reports"
FALSE_NEGATIVES_PATH = OUTPUTS_DIR / "false_negatives.csv"
FALSE_POSITIVES_PATH = OUTPUTS_DIR / "false_positives.csv"
BASELINE_METRICS_PATH = METRICS_DIR / "baseline_metrics.json"
TUNED_MODELS_SUMMARY_PATH = METRICS_DIR / "tuned_models_summary.json"

# Artifacts Paths
ARTIFACTS_DIR = BASE_DIR / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
MODEL_METADATA_DIR = ARTIFACTS_DIR / "model_metadata"
PREPROCESSING_SUMMARY_PATH = ARTIFACTS_DIR / "preprocessing_summary_report.json"
PRODUCTION_MODEL_REPORT_PATH = ARTIFACTS_DIR / "production_model_report.json"
REGISTRY_REPORT_PATH = ARTIFACTS_DIR / "registry_report.json"
REPRODUCIBILITY_REPORT_PATH = ARTIFACTS_DIR / "reproducibility_report.json"

# Logs
LOGS_DIR = BASE_DIR / "logs"
LOG_FILE_PATH = LOGS_DIR / "pipeline.log"

# Experiment Tracking
MLRUNS_DIR = BASE_DIR / "mlruns"
MLFLOW_DB_PATH = BASE_DIR / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
try:
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
except Exception:
    pass

MLFLOW_EXPERIMENT_NAME = "Amazon_Bestseller_Genre_Prediction"
REGISTERED_MODEL_NAME = "Amazon_Bestseller_Production_Model"

# Model & Split Configurations
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_SPLITS = 5
TARGET_COLUMN = "Genre"
RAW_FEATURE_COLUMNS = ["User Rating", "Reviews", "Price", "Year"]
ENGINEERED_FEATURE_COLUMNS = ["User Rating", "Reviews_Log", "Price", "Year"]
CLASS_NAMES = ["Fiction", "Non Fiction"]

# Ensure all essential directories exist
for path in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
    REPORTS_DIR,
    ARTIFACTS_DIR,
    PLOTS_DIR,
    MODEL_METADATA_DIR,
    LOGS_DIR,
    MLRUNS_DIR,
]:
    os.makedirs(str(path), exist_ok=True)
