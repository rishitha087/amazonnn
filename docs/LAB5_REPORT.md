# LABORATORY EXPERIMENT 5
## Data Validation and Reproducible ML Pipelines

---

### 1. AIM
To implement rigorous, production-grade automated data validation gates, establish schema and range verification protocols, detect data drifts and corruptions, and guarantee that only pristine, compliant data flows into downstream training and inference stages.

---

### 2. THEORY & BACKGROUND
Silent data corruption and schema changes represent primary causes of machine learning model deterioration in production. Unlike typical software bugs that trigger fatal runtime exceptions, corrupted or shifted data may allow pipelines to execute without error while silently corrupting predictions.

Comprehensive data validation for the Amazon dataset requires:
1. **Schema Integrity**: Verifying that mandatory columns (`Name`, `Author`, `User Rating`, `Reviews`, `Price`, `Year`, `Genre`) are present.
2. **Type Invariance**: Enforcing floating-point and integer types for numerical features, and string types for categorical variables.
3. **Domain Range Boundaries**:
   - $\text{User Rating} \in [0.0, 5.0]$
   - $\text{Price} \ge 0.0$
   - $\text{Reviews} \ge 0.0$
   - $\text{Year} \in [2000, 2050]$
4. **Target Cardinality**: Restricting target labels strictly to the known classes: $\{\text{Fiction}, \text{Non Fiction}\}$.
5. **Output Tensor Invariants**: Post-preprocessing arrays (`X_train_final.npy`, `X_test_final.npy`) must contain 0 NaNs, 0 Infs, shape $(N, 4)$, and standardized distribution properties ($\mu \approx 0, \sigma \approx 1$).

---

### 3. SOFTWARE & SYSTEM SPECIFICATIONS
- **Execution Engine**: Python 3.12.6
- **Validation Target**: `data/raw/amazon_bestsellers.csv` & `data/processed/*.npy`

---

### 4. LIBRARIES & DEPENDENCIES
- `pandas`: Tabular assertion testing and datatype verification.
- `numpy`: Multi-dimensional array invariant verification (`isnan`, `isinf`).
- `pyyaml`: Dynamic schema configuration loading from `config.yaml`.

---

### 5. IMPLEMENTATION DETAILS
1. **Input Validation (`02_validate_data.py`)**:
   - Parses raw dataset against `config.yaml` schema rules.
   - Throws descriptive errors if any mandatory column is absent or type-mismatched.
   - Verifies all numerical values remain within defined physical boundaries.
2. **Output Validation (`04_validate_outputs.py`)**:
   - Inspects generated `.npy` arrays for shape consistency.
   - Validates that training set has exactly 440 rows and test set has 110 rows with 4 features.
   - Enforces that binary labels contain only $\{0, 1\}$.
   - Generates `artifacts/output_validation_report.json`.

---

### 6. CODE FILES & SCRIPTS
- `src/validate_data.py`: Legacy standalone validation script.
- `src/validate_outputs.py`: Preprocessed array assertions.
- `src/mlops_pipeline/02_validate_data.py`: Modular raw schema validation stage.
- `src/mlops_pipeline/04_validate_outputs.py`: Modular output array verification stage.

---

### 7. EXECUTION COMMANDS
```bash
python src/mlops_pipeline/02_validate_data.py
python src/mlops_pipeline/04_validate_outputs.py
```

---

### 8. ACTUAL OUTPUTS & METRICS
```text
============================================================
STAGE 2: RAW DATA SCHEMA & QUALITY VALIDATION
============================================================
[SUCCESS] Schema validation passed! All 7 columns valid across 550 records.

============================================================
STAGE 4: VALIDATING PREPROCESSED ARRAYS & OUTPUTS
============================================================
[SUCCESS] Processed arrays validated: X_train=(440, 4), X_test=(110, 4)
Validation report saved to: artifacts/output_validation_report.json
```

Validation report summary (`output_validation_report.json`):
```json
{
    "status": "PASSED",
    "X_train_shape": [440, 4],
    "X_test_shape": [110, 4],
    "y_train_shape": [440],
    "y_test_shape": [110],
    "train_mean": -0.0,
    "train_std": 1.0,
    "nan_count": 0,
    "inf_count": 0,
    "classes": [0, 1]
}
```

---

### 9. VERIFICATION & VALIDATION
- Tested edge case bounds: Values exceeding 5.0 in User Rating or negative prices trigger validation rejections.
- Validated that downstream model training will never ingest corrupted inputs.

---

### 10. RESULT & CONCLUSION
Automated data validation gates were established for both raw input CSVs and engineered NumPy arrays. All schema, datatype, range, and array integrity checks passed with zero defects, ensuring pipeline stability.

