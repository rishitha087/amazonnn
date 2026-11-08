# LABORATORY EXPERIMENT 9
## Pipeline Failure Handling and Recovery

---

### 1. AIM
To construct an automated chaos engineering and failure injection testing suite, simulate adverse pipeline conditions across data schema, datatypes, boundaries, and model artifact availability, implement self-healing recovery mechanisms, and demonstrate 100% automated recovery capability.

---

### 2. THEORY & BACKGROUND
Resilience in production MLOps requires systems to gracefully detect and recover from operational failures without manual human intervention. Failures in production ML pipelines commonly stem from:
1. **Upstream Data Ingestion Failures**: Upstream ETL failures causing missing files or incomplete payloads.
2. **Schema Drift & Dropped Features**: Upstream changes dropping columns required by the model.
3. **Data Type Corruption**: Strings injected into numerical floating-point columns.
4. **Physical Range Violations**: Sensor or input errors generating out-of-domain values (e.g. negative prices, ratings $> 5.0$).
5. **Artifact Storage Failures**: Model or scaler files deleted or temporarily inaccessible.

The failure recovery system must guarantee:
- **Immediate Detection**: Catching anomalies before corrupt data reaches model inference or training.
- **Fail-Safe Self-Healing**: Automatically restoring from immutable snapshots (`.lab9_backup/`) or falling back to healthy replicas.
- **Audit Logging**: Capturing failure modes, detection latencies, recovery latencies, and post-recovery health metrics.

---

### 3. FAILURE INJECTION SCENARIOS
1. `SCENARIO_1_MISSING_RAW_DATASET`: Dataset file deleted or moved.
2. `SCENARIO_2_CORRUPT_COLUMN_SCHEMA`: Mandatory `Price` column dropped.
3. `SCENARIO_3_INVALID_DATATYPE_INJECTION`: Non-numeric string injected into `User Rating`.
4. `SCENARIO_4_OUT_OF_BOUNDS_VALUES`: User Rating set to `9.5` (allowed range: $0.0 \le \text{rating} \le 5.0$).
5. `SCENARIO_5_MISSING_MODEL_ARTIFACT`: Production model artifact `production_model.pkl` removed.

Each scenario is executed across **3 independent repetitions** ($5 \times 3 = 15$ test runs).

---

### 4. SOFTWARE & SYSTEM SPECIFICATIONS
- **Language**: Python 3.12.6
- **Backup Store**: `.lab9_backup/` immutable snapshot directory
- **Audit Target**: `reports/failure_report.json`

---

### 5. CODE FILES & SCRIPTS
- `src/mlops_pipeline/10_failure_injection.py`: Automated 15-run failure injection and self-healing test engine.
- `.github/workflows/lab9_failure_tests.yml`: Scheduled automated failure resilience workflow.

---

### 6. EXECUTION COMMANDS
```bash
python src/mlops_pipeline/10_failure_injection.py
```

---

### 7. ACTUAL OUTPUTS & METRICS
```text
============================================================
STAGE 10: LAB 9 PIPELINE FAILURE INJECTION & RECOVERY SUITE
============================================================
[SUCCESS] Lab 9 Failure Injection Suite Completed: 15/15 passed (100.0% Recovery Rate).
Report written to: reports/failure_report.json
```

Audit report summary (`failure_report.json`):
```json
{
    "suite": "Lab 9 Failure Injection and Automated Self-Healing",
    "total_scenarios": 5,
    "repetitions_per_scenario": 3,
    "total_test_runs": 15,
    "successful_recoveries": 15,
    "recovery_rate_percent": 100.0,
    "average_duration_ms": 142.3,
    "scenarios_evaluated": [
        "SCENARIO_1_MISSING_RAW_DATASET",
        "SCENARIO_2_CORRUPT_COLUMN_SCHEMA",
        "SCENARIO_3_INVALID_DATATYPE_INJECTION",
        "SCENARIO_4_OUT_OF_BOUNDS_VALUES",
        "SCENARIO_5_MISSING_MODEL_ARTIFACT"
    ]
}
```

---

### 8. VERIFICATION & VALIDATION
- Tested all 15 permutations. Every anomaly was identified with specific error types (`FileNotFoundError`, `SchemaError`, `TypeError`, `RangeError`, `ModelArtifactMissingError`).
- Verified that in 100% of cases, the clean backup was restored and post-recovery assertions passed.
- Confirmed that raw data and models were restored to their original pristine states following suite completion.

---

### 9. RESULT & CONCLUSION
The Lab 9 Failure Injection and Self-Healing suite executed 15 stress tests across 5 failure scenarios, achieving a **100.0% automated recovery rate** with an average detection and restoration latency of 142.3 ms.

