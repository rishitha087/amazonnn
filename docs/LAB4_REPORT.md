# LABORATORY EXPERIMENT 4
## Experiment Tracking and Reproducibility Using MLflow

---

### 1. AIM
To integrate an automated experiment tracking system using MLflow, log model hyperparameters, evaluation metrics, and diagnostic visualization artifacts, establish reproducible training pipelines, and guarantee deterministic evaluation across successive runs.

---

### 2. THEORY & BACKGROUND
Machine learning development is iterative and empirical. Without rigorous experiment tracking, models suffer from the "black box" reproducibility crisis, where hyperparameters, random seeds, and environment dependencies are lost.

MLflow provides an open-source platform managing the full ML lifecycle:
1. **MLflow Tracking**: An API and UI for logging parameters, code versions, metrics, and output files when running machine learning code.
2. **Backend Stores**:
   - Structured metadata (experiments, runs, metrics, params) stored in an enterprise relational store (`sqlite:///mlflow.db`).
   - Binary artifacts (model weights, plots, feature schemas) stored in a localized artifact repository (`mlruns/` or file store).
3. **Reproducibility Guarantee**: Enforcing fixed pseudo-random seeds (`random_state=42`), stratified sampling, and deterministic tree-building algorithms guarantees bit-for-bit identical evaluation metrics across consecutive executions.

---

### 3. SOFTWARE & SYSTEM SPECIFICATIONS
- **Tracking Backend**: SQLite database (`sqlite:///mlflow.db`)
- **Artifact Store**: Local filesystem (`mlruns/`)
- **Experiment Name**: `Amazon_Bestseller_Genre_Prediction`
- **Framework**: MLflow 3.16+ / scikit-learn 1.8+

---

### 4. LIBRARIES & DEPENDENCIES
- `mlflow` (>=2.8.0): Experiment tracking, fluent logging, and client API.
- `scikit-learn`: Estimator training and metrics calculation.
- `cloudpickle`: Safe binary object serialization for complex Python estimator pipelines.
- `matplotlib` & `seaborn`: Visualization artifact generation.

---

### 5. IMPLEMENTATION DETAILS
1. **Sanitized Tracking URI Configuration**:
   Configured normalized path resolution using `.as_posix()` without errant URL encoding to avoid stray `%20` directory generation on Windows:
   ```python
   db_path = BASE_DIR / "mlflow.db"
   tracking_uri = f"sqlite:///{db_path.as_posix()}"
   mlflow.set_tracking_uri(tracking_uri)
   ```
2. **Parameters Logged**:
   - `model_family`: "RandomForest"
   - `n_estimators`: 100
   - `max_depth`: 5 (or 10)
   - `random_state`: 42
3. **Metrics Logged**:
   - `accuracy`: 0.7909
   - `precision_weighted`: 0.7915
   - `recall_weighted`: 0.7909
   - `f1_score_weighted`: 0.7859
   - `roc_auc`: 0.8740
4. **Artifacts Logged**:
   - `plots/confusion_matrix.png`
   - `plots/roc_curve.png`
   - `metadata/dataset_metadata.json`
   - Model binary with scikit-learn flavor and cloudpickle serialization format.

---

### 6. CODE FILES & SCRIPTS
- `src/config.py`: Centralized MLflow tracking URI and experiment definitions.
- `src/train_mlflow.py`: Full multi-model experimentation suite with GridSearchCV logging.
- `src/validate_reproducibility.py`: Deterministic pipeline verification.
- `src/mlops_pipeline/05_train_registry.py`: Automated pipeline training and tracking run execution.

---

### 7. EXECUTION COMMANDS
```bash
python src/train_mlflow.py
python src/validate_reproducibility.py
```

---

### 8. ACTUAL OUTPUTS & METRICS
```text
============================================================
STAGE 5: MODEL TRAINING & MLFLOW REGISTRY
============================================================
[INFO] Experiment 'Amazon_Bestseller_Genre_Prediction' active.
[INFO] Logging parameters: {'n_estimators': 100, 'max_depth': 5, 'random_state': 42}
[INFO] Logging metrics:
  - Accuracy:           0.7909
  - Precision Weighted: 0.7915
  - Recall Weighted:    0.7909
  - F1-Score Weighted:  0.7859
  - ROC-AUC:            0.8740
[SUCCESS] Model trained & registered. Run ID: a06d784a4d00470ab2afcc5ebad1717f
```

Reproducibility validation result:
```text
============================================================
REPRODUCIBILITY VALIDATION SUITE
============================================================
Run 1 F1-Score: 0.785909
Run 2 F1-Score: 0.785909
Run 3 F1-Score: 0.785909
Delta: 0.0000000000000000
[PASS] Deterministic reproducibility verified!
```

---

### 9. VERIFICATION & VALIDATION
1. **Database Persistence**: Queried `sqlite:///mlflow.db` to verify runs, params, and metrics tables contain the recorded run ID `a06d784a4d00470ab2afcc5ebad1717f`.
2. **Artifact Integrity**: Confirmed plots and serialized models exist in the MLflow artifact location without corruption.
3. **Isolated File Structure**: Verified no root `%20` or stray database files were generated.

---

### 10. RESULT & CONCLUSION
MLflow experiment tracking was successfully established with SQLite persistence. All model hyperparameter variations, performance metrics, and graphical artifacts were systematically recorded, and deterministic 100% reproducibility was empirically proven.

