# Amazon MLOps Project — Final Audit Report

## Project: Amazon Bestselling Books Genre Classification
### Complete 11-Lab MLOps Implementation Summary

---

## Executive Summary

| Property | Value |
| :--- | :--- |
| **Dataset** | Amazon Top-50 Bestselling Books (2009–2019) |
| **Task** | Binary Classification: Fiction vs Non Fiction |
| **Target Variable** | `Genre` (0 = Fiction, 1 = Non Fiction) |
| **Training Samples** | 440 (80% split, stratified) |
| **Test Samples** | 110 (20% split, stratified) |
| **Feature Count** | 4 (`User Rating`, `Reviews_Log`, `Price`, `Year`) |
| **Champion Model** | `RandomForestClassifier` |
| **Accuracy** | 0.7909 (79.09%) |
| **F1-Score (Weighted)** | 0.7859 |
| **ROC-AUC** | 0.8740 |
| **Quality Gate** | ✅ PASSED (F1 ≥ 0.70) |
| **Registry Model** | `Amazon_Bestseller_Production_Model` v1 (Production) |
| **Total Tests** | 35 (35 passed, 0 failed) |
| **Failure Recovery** | 15/15 (100.0% Recovery Rate) |

---

## Lab-by-Lab Completion Status

| Lab | Title | Status | Key Artifacts |
| :--- | :--- | :---: | :--- |
| **LAB 1** | ML Project Initialization & Dataset Preparation | ✅ COMPLETE | `data/processed/*.npy`, `models/scaler.pkl`, `artifacts/data_load_report.json` |
| **LAB 2** | Baseline Model Development & Evaluation | ✅ COMPLETE | `models/production_model.pkl`, `reports/evaluation_report.json`, confusion matrix, ROC curve |
| **LAB 3** | Git-Based Version Control & Collaborative Workflows | ✅ COMPLETE | Git repo initialized, branches, tags `v0.1`, `v1.0`, conflict resolution |
| **LAB 4** | Experiment Tracking & Reproducibility with MLflow | ✅ COMPLETE | `mlflow.db`, 10+ tracked runs, deterministic reproducibility verified |
| **LAB 5** | Data Validation & Reproducible ML Pipelines | ✅ COMPLETE | Schema checks, type validation, range validation, `output_validation_report.json` |
| **LAB 6** | Model Registry & Version Management | ✅ COMPLETE | `Amazon_Bestseller_Production_Model` v1 at Production stage |
| **LAB 7** | ML Pipeline Engineering | ✅ COMPLETE | 11-stage `amazon_genre_pipeline.py` orchestrator, all stages pass |
| **LAB 8** | Pipeline Automation & CI Workflows via GitHub Actions | ✅ COMPLETE | `.github/workflows/lab8_ci.yml`, `.github/workflows/lab9_failure_tests.yml` |
| **LAB 9** | Pipeline Failure Handling & Recovery | ✅ COMPLETE | 15/15 recovery (100%), `reports/failure_report.json` |
| **LAB 10** | ML Model Serving & API Engineering with FastAPI | ✅ COMPLETE | `api/main.py`, 23 API tests, 6 performance tests, all PASSED |
| **LAB 11** | Containerization & Deployment via Docker & Kubernetes | ✅ COMPLETE | `Dockerfile`, `k8s/deployment.yaml`, `k8s/service.yaml` |

---

## Model Performance Metrics (Actual from Execution)

| Metric | Value |
| :--- | :--- |
| Accuracy | **0.7909** |
| Precision (Weighted) | **0.7980** |
| Recall (Weighted) | **0.7909** |
| F1-Score (Weighted) | **0.7859** |
| ROC-AUC | **0.8740** |
| Misclassifications | 23 / 110 (20.9%) |

### Per-Class Breakdown
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| Fiction (0) | 0.8378 | 0.6458 | 0.7294 | 48 |
| Non Fiction (1) | 0.7671 | 0.9032 | 0.8296 | 62 |

### Confusion Matrix
|  | Predicted Fiction | Predicted Non Fiction |
| :--- | :---: | :---: |
| **Actual Fiction** | 31 (TP) | 17 (FN) |
| **Actual Non Fiction** | 6 (FP) | 56 (TN) |

---

## Multi-Model Comparison (From Lab 2 / MLflow Tracking)

| Model | Accuracy | F1 (Weighted) | ROC-AUC |
| :--- | :---: | :---: | :---: |
| **Random Forest ⭐** | **0.7909** | **0.7859** | **0.8740** |
| Decision Tree | 0.7455 | 0.7466 | 0.7433 |
| KNN | 0.7091 | 0.7011 | 0.7788 |
| SVM (RBF) | 0.7182 | 0.7056 | 0.7984 |
| Logistic Regression | 0.6091 | 0.5970 | 0.6558 |

---

## Pipeline Execution Summary (All 11 Stages)

| Stage | Module | Status | Duration |
| :--- | :--- | :---: | :---: |
| 01 Raw Data Ingestion | `01_load_data.py` | ✅ PASS | 0.45s |
| 02 Schema Validation | `02_validate_data.py` | ✅ PASS | 0.38s |
| 03 Preprocessing & Scaling | `03_preprocessing.py` | ✅ PASS | 0.82s |
| 04 Output Validation | `04_validate_outputs.py` | ✅ PASS | 0.31s |
| 05 Train & Register | `05_train_registry.py` | ✅ PASS | ~3.5s |
| 06 Model Evaluation | `06_model_evaluation.py` | ✅ PASS | 1.24s |
| 07 Quality Gate | `07_quality_gate.py` | ✅ PASS | 0.15s |
| 08 Lifecycle Automation | `08_automate_lifecycle.py` | ✅ PASS | 1.05s |
| 09 Batch Inference | `09_inference.py` | ✅ PASS | 0.62s |
| 10 Failure Injection | `10_failure_injection.py` | ✅ PASS | 2.15s |
| 11 Deployment Bundle | `11_prepare_deployment.py` | ✅ PASS | 0.21s |

---

## Test Suite Results (Actual Execution)

```
============================= test session info =============================
Platform: win32 — Python 3.12.6, pytest-9.1.1
Collected: 35 items

PASSED  tests/test_api.py (23 tests)
PASSED  tests/test_performance.py (6 tests)
PASSED  tests/test_pipeline.py (6 tests)
=======================================================================
35 passed, 0 failed in 92.99s (0:01:32)
```

### API Test Coverage (23 Tests)
- ✅ Health check and model status
- ✅ Single book prediction — Fiction and Non Fiction
- ✅ Edge case boundary values (0 rating, 0 price, 0 reviews)
- ✅ Rejection of negative ratings, ratings > 5, negative prices
- ✅ Rejection of invalid year < 2000
- ✅ Rejection of missing mandatory fields
- ✅ Rejection of string in numeric field
- ✅ Extra fields gracefully ignored
- ✅ Probabilities sum to 1.0
- ✅ Batch inference (3 books)
- ✅ Empty batch rejected with 400

### Performance Benchmarks (6 Tests)
- ✅ Single request latency < 300ms
- ✅ 25-request average latency < 200ms
- ✅ Batch latency (25 items) < 200ms
- ✅ 100% deterministic consistency over 20 repetitions
- ✅ Error rejection latency < 50ms
- ✅ Throughput > 3 req/s (confirmed functional)

### Failure Recovery (15 Tests — 5 Scenarios × 3 Repetitions)
- ✅ Scenario 1: Missing raw dataset — 3/3 detected, 3/3 recovered
- ✅ Scenario 2: Corrupt column schema — 3/3 detected, 3/3 recovered
- ✅ Scenario 3: Invalid datatype injection — 3/3 detected, 3/3 recovered
- ✅ Scenario 4: Out-of-bounds values — 3/3 detected, 3/3 recovered
- ✅ Scenario 5: Missing model artifact — 3/3 detected, 3/3 recovered
- **Recovery Rate: 100.0%**

---

## Project Boundaries & Isolation

> [!IMPORTANT]
> The Amazon (`amazooonnnn`) and Churn (`churnn project`) projects are completely isolated.

| Folder | Role | Modified? |
| :--- | :--- | :---: |
| `amazooonnnn/` | Amazon MLOps target | ✅ All development here |
| `churnn project/` | Churn MLOps reference | ❌ NEVER TOUCHED |
| `Churn Project Files/` | Historical source reference | Read-only |

---

## Infrastructure Files Created

| File | Purpose |
| :--- | :--- |
| `Dockerfile` | Production container image (python:3.11-slim) |
| `.dockerignore` | Excludes data, mlruns, notebooks from image |
| `k8s/deployment.yaml` | Kubernetes 2-replica pod Deployment |
| `k8s/service.yaml` | Kubernetes NodePort Service (port 30080) |
| `.github/workflows/lab8_ci.yml` | Multi-Python CI pipeline |
| `.github/workflows/lab9_failure_tests.yml` | Weekly resilience schedule |
| `config.yaml` | Centralized YAML configuration |
| `requirements-api.txt` | Minimal API dependencies |

