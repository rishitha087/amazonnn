# Amazon Bestselling Books — MLOps Labs 1–6 Project Guide

## Overview
This project converts an existing Amazon Bestselling Books ML notebook into a
production-grade MLOps pipeline, following a 6-lab progression from raw data
preprocessing through model registry automation.

**Dataset**: Amazon Top 50 Bestselling Books 2009-2019
**Task**: Binary classification — Fiction vs Non Fiction
**Features**: User Rating, Reviews (log-transformed), Price, Year

---

## Lab 1 — Data Preprocessing & Dataset Preparation

**Script**: `src/preprocess.py`

| Step | Description |
|------|-------------|
| 1 | Load raw CSV (550 rows × 7 columns) |
| 2 | Handle missing values (none present in this dataset) |
| 3 | Remove exact duplicates |
| 4 | Encode target: Genre → Fiction=0, Non Fiction=1 |
| 5 | Engineer Reviews_Log = log1p(Reviews) |
| 6 | Select features: User Rating, Reviews_Log, Price, Year |
| 7 | Train/test split (80/20, stratified, random_state=42) |
| 8 | Fit StandardScaler on train, transform both sets |
| 9 | Save X_train.npy, X_test.npy, y_train.npy, y_test.npy |
| 10 | Save scaler.pkl, encoder.pkl, preprocessor.pkl |

**Run**: `python -m src.preprocess`

---

## Lab 2 — ML Model Experimentation & Evaluation

**Scripts**: `src/train.py`, `src/evaluate.py`

- Baseline model: RandomForestClassifier (n_estimators=100, random_state=42)
- Evaluation metrics: Accuracy, Precision, Recall, F1 (all weighted), ROC-AUC
- Confusion matrix visualization saved as PNG
- Error analysis: False Positives and False Negatives exported to CSV

**Run**: `python -m src.train` then `python -m src.evaluate`

---

## Lab 3 — Baseline ML Pipeline

**Script**: `pipelines/run_lab3_baseline.py`

Orchestrates the end-to-end baseline pipeline:
1. Preprocess (Lab 1)
2. Train (Lab 2)
3. Evaluate (Lab 2)
4. Predict (inference demo)

**Run**: `python -m pipelines.run_lab3_baseline`

---

## Lab 4 — Experiment Tracking & Reproducibility (MLflow)

**Scripts**: `src/train_mlflow.py`, `src/validate_reproducibility.py`
**Pipeline**: `pipelines/run_lab4_tracking.py`

### Multi-Algorithm Experiment Tracking
Tracks 5 algorithms with hyperparameter grid search (GridSearchCV, 5-fold CV):

| Algorithm | Key Hyperparameters |
|-----------|-------------------|
| Logistic Regression | C, solver, max_iter |
| K-Nearest Neighbors | n_neighbors, weights, metric |
| Decision Tree | max_depth, min_samples_split, min_samples_leaf |
| Random Forest | n_estimators, max_depth, min_samples_split |
| Gradient Boosting | n_estimators, learning_rate, max_depth |

Each experiment logs:
- Parameters, metrics (accuracy, precision, recall, f1, roc_auc)
- Confusion matrix artifact
- Best model artifact via MLflow

### Reproducibility Verification
- Fixed random seeds (RANDOM_STATE=42)
- Dual-run deterministic comparison
- Report saved to `artifacts/reproducibility_report.json`

**Run**: `python -m pipelines.run_lab4_tracking`
**View**: `mlflow ui --backend-store-uri mlruns`

---

## Lab 5 — Production Data Pipeline & Schema Validation

**Scripts**: `src/preprocess_pipeline.py`, `src/validate_data.py`, `src/validate_outputs.py`
**Pipeline**: `pipelines/run_lab5_pipeline.py`

### Sklearn Pipeline
- ColumnTransformer with StandardScaler → RandomForest
- Serialized pipeline artifact for reproducible deployment

### Data Quality Validation (Great Expectations)
- Column presence verification (Name, Author, User Rating, Reviews, Price, Year, Genre)
- Type checking (Rating as float, Reviews/Price as int, Year as int)
- Value range validation (Rating 0-5, Price 0-200, Year 2009-2030)
- Genre domain validation (Fiction, Non Fiction)
- Null checks on all columns

### Output Artifact Validation
- Processed array existence and shape verification
- Model serialization integrity check
- Metrics JSON completeness verification

**Run**: `python -m pipelines.run_lab5_pipeline`

---

## Lab 6 — MLflow Model Registry & Champion Lifecycle

**Scripts**: `src/train_registry.py`, `src/automate_lifecycle.py`, `src/generate_registry_report.py`
**Pipeline**: `pipelines/run_lab6_registry.py`

### Model Registry
- Registers trained model under `Amazon_Bestseller_Production_Model`
- Logs all metrics, parameters, and preprocessing artifacts

### Automated Lifecycle Management
- Stage transitions: None → Staging → Production → Archived
- Champion/Challenger comparison on `f1_score_weighted`
- Automatic promotion if challenger >= champion

### Deployment Report
- JSON reports: `artifacts/registry_report.json`, `artifacts/production_model_report.json`
- Includes model metadata, metrics, version history

**Run**: `python -m pipelines.run_lab6_registry`

---

## Production Inference

**Script**: `src/predict.py`

The `AmazonBestsellerPredictor` class provides:
- Single-book prediction via dictionary input
- Batch prediction via DataFrame input
- Automatic feature engineering (Reviews_Log)
- Confidence scores with probability estimates

**Example**:
```python
from src.predict import AmazonBestsellerPredictor

predictor = AmazonBestsellerPredictor()
result = predictor.predict({
    "User Rating": 4.7,
    "Reviews": 15000,
    "Price": 15,
    "Year": 2020
})
print(result)
# [{'predicted_genre': 'Non Fiction', 'confidence': 0.73, ...}]
```

---

## Running Everything

```bash
# Complete pipeline (all labs + prediction)
python -m pipelines.run_all

# Tests (6 tests)
python -m pytest tests/ -v

# MLflow dashboard
mlflow ui --backend-store-uri mlruns
```
