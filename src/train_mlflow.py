import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn

from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay, classification_report
)

try:
    from src.config import (
        X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH,
        DATASET_METADATA_PATH, RANDOM_FOREST_MODEL_PATH,
        PRODUCTION_MODEL_PATH, NOTEBOOK_SAVED_MODEL_PATH,
        ARTIFACTS_DIR, PLOTS_DIR, OUTPUTS_DIR, METRICS_DIR,
        TUNED_MODELS_SUMMARY_PATH, MLFLOW_EXPERIMENT_NAME,
        MLFLOW_TRACKING_URI, RANDOM_STATE, CV_SPLITS, CLASS_NAMES
    )
    from src.utils import get_logger, save_json
    logger = get_logger("Train_MLflow_Lab4")
except ImportError:
    BASE_DIR = Path(__file__).resolve().parent.parent
    PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
    MODELS_DIR = BASE_DIR / "models"
    OUTPUTS_DIR = BASE_DIR / "outputs"
    METRICS_DIR = OUTPUTS_DIR / "metrics"
    ARTIFACTS_DIR = BASE_DIR / "artifacts"
    PLOTS_DIR = ARTIFACTS_DIR / "plots"
    X_TRAIN_PATH = PROCESSED_DATA_DIR / "X_train_final.npy"
    X_TEST_PATH = PROCESSED_DATA_DIR / "X_test_final.npy"
    Y_TRAIN_PATH = PROCESSED_DATA_DIR / "y_train.npy"
    Y_TEST_PATH = PROCESSED_DATA_DIR / "y_test.npy"
    DATASET_METADATA_PATH = PROCESSED_DATA_DIR / "dataset_metadata.json"
    RANDOM_FOREST_MODEL_PATH = MODELS_DIR / "random_forest_model.pkl"
    PRODUCTION_MODEL_PATH = MODELS_DIR / "production_model.pkl"
    NOTEBOOK_SAVED_MODEL_PATH = BASE_DIR / "bestseller_genre_model.pkl"
    TUNED_MODELS_SUMMARY_PATH = METRICS_DIR / "tuned_models_summary.json"
    MLFLOW_EXPERIMENT_NAME = "Amazon_Bestseller_Genre_Prediction"
    MLFLOW_TRACKING_URI = (BASE_DIR / "mlruns").as_uri()
    RANDOM_STATE = 42
    CV_SPLITS = 5
    CLASS_NAMES = ["Fiction", "Non Fiction"]
    import logging
    logger = logging.getLogger("Train_MLflow_Lab4")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    def save_json(d, p):
        import json
        with open(str(p), "w") as f:
            json.dump(d, f, indent=4)


def train_and_track_run(model, run_name: str, model_family: str, version_type: str,
                         params: dict, X_train, y_train, X_test, y_test, cv):
    """Trains a single model variant, runs cross-validation, generates plots, and logs to MLflow."""
    logger.info(f"\n---> MLflow Run: {run_name} [{model_family} | {version_type}]")

    # Fit model
    model.fit(X_train, y_train)

    # 5-fold Stratified Cross-Validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_weighted")
    cv_mean = float(cv_scores.mean())
    cv_std = float(cv_scores.std())

    # Predict on test partition
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # Calculate metrics matching notebook (weighted averaging)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_score_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "cv_f1_mean": cv_mean,
        "cv_f1_std": cv_std
    }
    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))

    # Log to MLflow
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("model_family", model_family)
        mlflow.log_param("version_type", version_type)
        mlflow.log_param("random_state", RANDOM_STATE)
        
        # Flatten parameters for clean logging
        clean_params = {k: (str(v) if isinstance(v, (list, tuple, dict)) else v) for k, v in params.items()}
        mlflow.log_params(clean_params)

        mlflow.log_metrics(metrics)

        # 1. Confusion Matrix Plot
        fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax_cm)
        ax_cm.set_title(f"{model_family} - {version_type} Confusion Matrix")
        ax_cm.set_xlabel("Predicted Label")
        ax_cm.set_ylabel("Actual Label")
        plt.tight_layout()

        cm_filename = f"confusion_matrix_{model_family}_{version_type}.png"
        cm_path = PLOTS_DIR / cm_filename
        fig_cm.savefig(str(cm_path), bbox_inches="tight")
        plt.close(fig_cm)
        mlflow.log_artifact(str(cm_path), artifact_path="plots")

        # 2. ROC Curve Plot (if probabilities available)
        if y_proba is not None:
            fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
            RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax_roc, name=f"{model_family} {version_type}")
            ax_roc.set_title(f"ROC Curve - {model_family} {version_type}")
            plt.tight_layout()
            roc_filename = f"roc_curve_{model_family}_{version_type}.png"
            roc_path = PLOTS_DIR / roc_filename
            fig_roc.savefig(str(roc_path), bbox_inches="tight")
            plt.close(fig_roc)
            mlflow.log_artifact(str(roc_path), artifact_path="plots")

        # 3. Log Dataset Metadata
        if os.path.exists(str(DATASET_METADATA_PATH)):
            mlflow.log_artifact(str(DATASET_METADATA_PATH), artifact_path="metadata")

        # 4. Log Model Flavor
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
        )

        logger.info(
            f"Logged Run '{run_name}': Test F1={metrics['f1_score_weighted']:.4f} | "
            f"CV F1 Mean={metrics['cv_f1_mean']:.4f} | Accuracy={metrics['accuracy']:.4f}"
        )

    return metrics, model, cm_path


def run_all_experiments():
    """Executes full model experimentation (Lab 2) and MLflow tracking suite (Lab 4)
    for all 5 models from the Amazon notebook.
    """
    logger.info("Executing Complete Amazon MLflow Experimentation Suite (Lab 2 & Lab 4)...")
    os.makedirs(str(PLOTS_DIR), exist_ok=True)
    os.makedirs(str(METRICS_DIR), exist_ok=True)

    # 1. Load Preprocessed Data
    X_train = np.load(str(X_TRAIN_PATH))
    X_test = np.load(str(X_TEST_PATH))
    y_train = np.load(str(Y_TRAIN_PATH))
    y_test = np.load(str(Y_TEST_PATH))

    # Cross-validation strategy matching notebook
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # Set MLflow Experiment
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # 2. Define Model Configurations & Hyperparameter Grids (from amazonnn.ipynb)
    models_config = [
        {
            "family": "LogisticRegression",
            "baseline_model": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            "baseline_params": {"max_iter": 1000, "random_state": RANDOM_STATE},
            "param_grid": {
                "C": [0.01, 0.1, 1, 10, 100],
                "solver": ["liblinear", "lbfgs"]
            },
            "grid_estimator": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        },
        {
            "family": "KNN",
            "baseline_model": KNeighborsClassifier(n_neighbors=5),
            "baseline_params": {"n_neighbors": 5},
            "param_grid": {
                "n_neighbors": [3, 5, 7, 9, 11, 15],
                "weights": ["uniform", "distance"],
                "p": [1, 2]
            },
            "grid_estimator": KNeighborsClassifier()
        },
        {
            "family": "DecisionTree",
            "baseline_model": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "baseline_params": {"random_state": RANDOM_STATE},
            "param_grid": {
                "criterion": ["gini", "entropy"],
                "max_depth": [None, 3, 5, 7, 10],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            },
            "grid_estimator": DecisionTreeClassifier(random_state=RANDOM_STATE)
        },
        {
            "family": "RandomForest",
            "baseline_model": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
            "baseline_params": {"n_estimators": 200, "random_state": RANDOM_STATE},
            "param_grid": {
                "n_estimators": [100, 200],
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2]
            },
            "grid_estimator": RandomForestClassifier(random_state=RANDOM_STATE)
        },
        {
            "family": "SVM",
            "baseline_model": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
            "baseline_params": {"kernel": "rbf", "probability": True, "random_state": RANDOM_STATE},
            "param_grid": {
                "C": [0.1, 1, 10, 100],
                "kernel": ["linear", "rbf"],
                "gamma": ["scale", "auto"]
            },
            "grid_estimator": SVC(probability=True, random_state=RANDOM_STATE)
        }
    ]

    all_results = []
    trained_tuned_models = {}

    for cfg in models_config:
        family = cfg["family"]
        logger.info(f"\n==================== EXPERIMENTING: {family} ====================")

        # A. Baseline Run
        base_name = f"{family}_Baseline"
        base_metrics, base_model, _ = train_and_track_run(
            model=cfg["baseline_model"],
            run_name=base_name,
            model_family=family,
            version_type="Baseline",
            params=cfg["baseline_params"],
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            cv=cv
        )
        all_results.append({
            "Model": family,
            "Version": "Baseline",
            **base_metrics
        })

        # B. Hyperparameter Tuning via GridSearchCV
        logger.info(f"Tuning {family} with GridSearchCV over {len(cfg['param_grid'])} param dimensions...")
        grid = GridSearchCV(
            estimator=cfg["grid_estimator"],
            param_grid=cfg["param_grid"],
            scoring="f1_weighted",
            cv=cv,
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        best_params = grid.best_params_
        best_estimator = grid.best_estimator_
        logger.info(f"Best parameters for {family}: {best_params} (Best CV F1: {grid.best_score_:.4f})")

        # C. Tuned Run
        tuned_name = f"{family}_Tuned"
        tuned_metrics, tuned_model, _ = train_and_track_run(
            model=best_estimator,
            run_name=tuned_name,
            model_family=family,
            version_type="Tuned",
            params=best_params,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            cv=cv
        )
        all_results.append({
            "Model": family,
            "Version": "Tuned",
            **tuned_metrics
        })
        trained_tuned_models[family] = tuned_model

        # D. Baseline vs Tuned Comparison Plot for this family
        fig_comp, ax_comp = plt.subplots(figsize=(8, 4))
        df_family = pd.DataFrame([
            {"Version": "Baseline", "F1 Score": base_metrics["f1_score_weighted"], "CV F1": base_metrics["cv_f1_mean"], "Accuracy": base_metrics["accuracy"]},
            {"Version": "Tuned", "F1 Score": tuned_metrics["f1_score_weighted"], "CV F1": tuned_metrics["cv_f1_mean"], "Accuracy": tuned_metrics["accuracy"]}
        ]).set_index("Version")
        df_family.plot(kind="bar", ax=ax_comp, ylim=(0, 1), rot=0)
        ax_comp.set_title(f"{family} - Baseline vs Tuned Performance")
        ax_comp.set_ylabel("Score")
        plt.tight_layout()
        comp_path = PLOTS_DIR / f"comparison_{family}.png"
        fig_comp.savefig(str(comp_path), bbox_inches="tight")
        plt.close(fig_comp)

    # 3. Overall Tuned Comparison (matching notebook Cell 113 & 115)
    df_all = pd.DataFrame(all_results)
    tuned_summary = df_all[df_all["Version"] == "Tuned"].sort_values(
        by=["cv_f1_mean", "f1_score_weighted"],
        ascending=False
    ).reset_index(drop=True)

    # Plot Model Comparison Bar Chart
    fig_overall, ax_overall = plt.subplots(figsize=(10, 5))
    plot_data = tuned_summary.set_index("Model")[["f1_score_weighted", "cv_f1_mean"]]
    plot_data.plot(kind="bar", ax=ax_overall, ylim=(0, 1), rot=30)
    ax_overall.set_title("Tuned Model Comparison - Weighted F1 Score")
    ax_overall.set_ylabel("Weighted F1 Score")
    ax_overall.legend(["Test F1 Score", "CV F1 Mean"])
    plt.tight_layout()
    overall_comp_path = PLOTS_DIR / "model_comparison_f1.png"
    fig_overall.savefig(str(overall_comp_path), bbox_inches="tight")
    plt.close(fig_overall)

    # Baseline vs Tuned Comparison Across all models
    fig_bvt, ax_bvt = plt.subplots(figsize=(10, 5))
    pivot_df = df_all.pivot(index="Model", columns="Version", values="f1_score_weighted")
    pivot_df.plot(kind="bar", ax=ax_bvt, ylim=(0, 1), rot=30)
    ax_bvt.set_title("All Models: Baseline vs Tuned Weighted F1")
    ax_bvt.set_ylabel("Weighted F1 Score")
    plt.tight_layout()
    bvt_comp_path = PLOTS_DIR / "baseline_vs_tuned_comparison.png"
    fig_bvt.savefig(str(bvt_comp_path), bbox_inches="tight")
    plt.close(fig_bvt)

    # 4. Final Best Model Selection
    best_model_name = tuned_summary.loc[0, "Model"]
    best_model = trained_tuned_models[best_model_name]
    logger.info(f"\n=======================================================")
    logger.info(f"🏆 OVERALL BEST MODEL: {best_model_name}")
    logger.info(f"Test F1: {tuned_summary.loc[0, 'f1_score_weighted']:.4f} | CV F1 Mean: {tuned_summary.loc[0, 'cv_f1_mean']:.4f}")
    logger.info(f"=======================================================\n")

    # 5. Save best models locally
    joblib.dump(best_model, str(PRODUCTION_MODEL_PATH))
    joblib.dump(best_model, str(RANDOM_FOREST_MODEL_PATH))
    joblib.dump(best_model, str(NOTEBOOK_SAVED_MODEL_PATH))
    logger.info(f"Saved production champion model to '{PRODUCTION_MODEL_PATH}' and '{NOTEBOOK_SAVED_MODEL_PATH}'")

    # 6. Save comprehensive summary metrics
    summary_data = {
        "best_model": best_model_name,
        "best_metrics": tuned_summary.iloc[0].to_dict(),
        "all_tuned_models": tuned_summary.to_dict(orient="records"),
        "all_experiments": df_all.to_dict(orient="records")
    }
    save_json(summary_data, TUNED_MODELS_SUMMARY_PATH)
    logger.info(f"Saved experiment summary to '{TUNED_MODELS_SUMMARY_PATH}'")

    return summary_data

if __name__ == "__main__":
    run_all_experiments()
