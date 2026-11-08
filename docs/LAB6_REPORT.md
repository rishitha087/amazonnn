# LABORATORY EXPERIMENT 6
## Model Registry and Version Management

---

### 1. AIM
To implement a centralized, auditable Model Registry using MLflow, manage model versioning, establish lifecycle stages (`None`, `Staging`, `Production`, `Archived`), and implement automated promotion gates comparing incoming candidate models against the active Production champion.

---

### 2. THEORY & BACKGROUND
In production environments, training a performant model is insufficient without formal release governance. The MLflow Model Registry provides a centralized model store, versioning mechanism, and stage transition engine:
1. **Centralized Metadata**: Maps model binaries to unique run IDs, Git commit SHA, training hyperparameters, and dataset snapshots.
2. **Version Semantics**: Sequential integer versioning (`v1`, `v2`, ...) automatically incremented upon model logging.
3. **Stage Transitions**:
   - `None`: Raw logged candidate.
   - `Staging`: Validated candidate undergoing acceptance and regression testing.
   - `Production`: Actively serving champion model servicing live inference traffic.
   - `Archived`: Deprecated historical models preserved for audit and rollbacks.
4. **Automated Champion-Challenger Lifecycle**:
   $$\text{Decision} = \begin{cases} \text{Promote to Production} & \text{if } F1_{\text{candidate}} \ge F1_{\text{champion}} \\ \text{Retain in Staging} & \text{otherwise} \end{cases}$$

---

### 3. SOFTWARE & SYSTEM SPECIFICATIONS
- **Registry Backend**: SQLite (`sqlite:///mlflow.db`)
- **Registered Model Name**: `Amazon_Bestseller_Production_Model`
- **Evaluation Metric**: `f1_score_weighted` (Quality Gate $\ge 0.70$)

---

### 4. LIBRARIES & DEPENDENCIES
- `mlflow` (>=2.8.0): Model Registry client API (`MlflowClient`).
- `scikit-learn`: Estimator persistence.
- `pyyaml`: Configuration threshold retrieval.

---

### 5. IMPLEMENTATION DETAILS
1. **Registration Protocol (`05_train_registry.py`)**:
   Logs trained Random Forest estimator with `registered_model_name="Amazon_Bestseller_Production_Model"` and cloudpickle serialization.
2. **Automated Lifecycle Stage Transition (`08_automate_lifecycle.py`)**:
   - Connects to MLflow Model Registry via `MlflowClient`.
   - Retrieves active Production champion version and metrics.
   - Retrieves latest Staging candidate version and metrics.
   - If no champion exists, promotes version 1 immediately to Production.
   - If candidate achieves higher or equal F1 score, automatically transitions candidate to `Production` and archives previous versions.
   - Exports decision summary to `reports/lifecycle_report.json`.

---

### 6. CODE FILES & SCRIPTS
- `src/train_registry.py`: Legacy standalone registry training script.
- `src/automate_lifecycle.py`: Standalone stage transition script.
- `src/mlops_pipeline/05_train_registry.py`: Modular training and registry registration.
- `src/mlops_pipeline/08_automate_lifecycle.py`: Automated candidate vs champion promotion gate.

---

### 7. EXECUTION COMMANDS
```bash
python src/mlops_pipeline/05_train_registry.py
python src/mlops_pipeline/08_automate_lifecycle.py
```

---

### 8. ACTUAL OUTPUTS & METRICS
```text
============================================================
STAGE 5: MODEL TRAINING & MLFLOW REGISTRY
============================================================
Successfully registered model 'Amazon_Bestseller_Production_Model'.
Created version '1' of model 'Amazon_Bestseller_Production_Model'.
[SUCCESS] Model trained & registered. Run ID: a06d784a4d00470ab2afcc5ebad1717f
Metrics: Accuracy=0.7909, F1-Weighted=0.7859, ROC-AUC=0.8740

============================================================
STAGE 8: MLFLOW MODEL LIFECYCLE AUTOMATION & STAGE TRANSITIONS
============================================================
[SUCCESS] Model Lifecycle transition complete: Promoted version 1 to Production (Initial Champion)
Report saved to: reports/lifecycle_report.json
```

`reports/lifecycle_report.json`:
```json
{
    "model_name": "Amazon_Bestseller_Production_Model",
    "action": "Promoted version 1 to Production (Initial Champion)",
    "active_production_version": "1"
}
```

---

### 9. VERIFICATION & VALIDATION
1. **Registry Verification**: Confirmed via `MlflowClient().get_latest_versions("Amazon_Bestseller_Production_Model")` that Version 1 has `current_stage="Production"`.
2. **Serving Integration**: Verified in Stage 9 that `mlflow.sklearn.load_model("models:/Amazon_Bestseller_Production_Model/Production")` successfully resolves and executes inference.

---

### 10. RESULT & CONCLUSION
The MLflow Model Registry was successfully configured. Model Version 1 of `Amazon_Bestseller_Production_Model` was verified, promoted to Production, and cataloged with full lineage and auditability.

