# Amazon Bestselling Books — MLOps Project (Labs 1–6)

A **production-grade MLOps pipeline** for classifying Amazon Top-50 Bestselling Books
(2009-2019) into **Fiction** vs **Non Fiction** genres.

Built following the same Lab 1–6 architecture as the Telco Customer Churn reference project,
but fully adapted to the Amazon Bestselling Books dataset and ML workflow.

---

## Dataset

| Column      | Description                                    |
|-------------|------------------------------------------------|
| Name        | Book title                                     |
| Author      | Author name                                    |
| User Rating | Average Amazon user rating (1-5)               |
| Reviews     | Total number of reviews                        |
| Price       | Price in USD                                   |
| Year        | Year the book appeared in the Top-50 list      |
| Genre       | **Target variable** — Fiction or Non Fiction   |

**Source**: [Kaggle — Amazon Top 50 Bestselling Books (2009-2019)](https://www.kaggle.com/datasets/sootersaalu/amazon-top-50-bestselling-books-2009-2019)

---

## Project Structure

```
amazooonnnn/
├── data/
│   ├── raw/                 # Raw CSV dataset
│   └── processed/           # Scaled X_train/X_test, encoded y_train/y_test, scaler
├── models/                  # Serialized baseline model (.pkl)
├── artifacts/               # Pipeline artifacts, registry reports
├── outputs/                 # Evaluation metrics, error analysis CSVs
├── mlruns/                  # MLflow tracking data
├── src/
│   ├── config.py            # Centralized paths, hyperparameters, feature lists
│   ├── utils.py             # Logging & seed utilities
│   ├── preprocess.py        # Lab 1: Data cleaning, feature engineering, scaling
│   ├── train.py             # Lab 2: Baseline RandomForest training
│   ├── evaluate.py          # Lab 2: Evaluation metrics, confusion matrix, error analysis
│   ├── train_mlflow.py      # Lab 4: Multi-algorithm MLflow experiment tracking
│   ├── validate_reproducibility.py  # Lab 4: Deterministic pipeline verification
│   ├── preprocess_pipeline.py       # Lab 5: Sklearn Pipeline with schema validation
│   ├── validate_data.py     # Lab 5: Great Expectations data quality checks
│   ├── validate_outputs.py  # Lab 5: Output artifact integrity validation
│   ├── train_registry.py    # Lab 6: Model training + MLflow Registry registration
│   ├── automate_lifecycle.py        # Lab 6: Champion/Challenger lifecycle automation
│   ├── generate_registry_report.py  # Lab 6: Production deployment report generation
│   └── predict.py           # Production inference engine
├── pipelines/
│   ├── run_lab3_baseline.py          # Lab 3 orchestrator
│   ├── run_lab4_tracking.py          # Lab 4 orchestrator
│   ├── run_lab5_pipeline.py          # Lab 5 orchestrator
│   ├── run_lab6_registry.py          # Lab 6 orchestrator
│   └── run_all.py                    # Master pipeline (all labs)
├── tests/
│   └── test_pipeline.py              # Pytest suite (6 tests)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Lab Breakdown

### Lab 1 — Data Preprocessing & Dataset Preparation
- Load raw CSV, handle missing values, encode `Genre` target (Fiction=1, Non Fiction=0)
- Engineer `Reviews_Log` feature (log-transform of Reviews)
- Feature set: `User Rating`, `Reviews_Log`, `Price`
- StandardScaler fitting and train/test split (80/20)
- Save processed NumPy arrays + scaler artifact

### Lab 2 — ML Model Experimentation & Evaluation
- Baseline RandomForest classifier (n_estimators=100, random_state=42)
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
- Confusion matrix, classification report
- Error analysis: False Positives / False Negatives export

### Lab 3 — Baseline ML Pipeline
- End-to-end orchestration: preprocess → train → evaluate → predict
- Single command execution

### Lab 4 — Experiment Tracking & Reproducibility (MLflow)
- 5 algorithms tracked: LogisticRegression, KNN, DecisionTree, RandomForest, GradientBoosting
- Each with hyperparameter grid search (GridSearchCV, 5-fold CV)
- All metrics, params, and artifacts logged to MLflow
- Deterministic reproducibility verification

### Lab 5 — Production Data Pipeline & Validation
- Sklearn Pipeline (StandardScaler → RandomForest) serialization
- Great Expectations data quality validation
- Schema validation (column types, value ranges)
- Output artifact integrity checks

### Lab 6 — MLflow Model Registry & Lifecycle
- Model registration in MLflow Model Registry
- Automated Champion/Challenger comparison
- Stage transitions: None → Staging → Production → Archived
- Production deployment report generation

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the complete pipeline (Labs 3-6 + Prediction)
python -m pipelines.run_all

# Or run individual labs:
python -m pipelines.run_lab3_baseline
python -m pipelines.run_lab4_tracking
python -m pipelines.run_lab5_pipeline
python -m pipelines.run_lab6_registry

# Run tests
pytest tests/ -v

# Launch MLflow UI (to view experiments)
mlflow ui --backend-store-uri mlruns
```

---

## Requirements

- Python 3.10+
- scikit-learn, pandas, numpy, matplotlib, seaborn
- mlflow, joblib, great_expectations, pytest
