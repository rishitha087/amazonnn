# LABORATORY EXPERIMENT 8
## Pipeline Automation and CI Workflows Using GitHub Actions

---

### 1. AIM
To construct and configure an automated Continuous Integration (CI) workflow using GitHub Actions, automate unit, integration, and performance testing, trigger end-to-end pipeline executions on code commits, and archive generated model and evaluation artifacts.

---

### 2. THEORY & BACKGROUND
Continuous Integration (CI) in MLOps enforces code quality, prevents regressions, and guarantees that every code change (push, pull request) maintains end-to-end system viability. Unlike traditional CI focused purely on code compilation and unit tests, MLOps CI must additionally validate:
1. **Pipeline Determinism**: Ensuring that training scripts yield expected performance metrics.
2. **Data & Schema Integrity**: Verifying that data validation passes against strict schemas.
3. **Serving API Compatibility**: Validating REST endpoints, status codes, latency budgets, and payload validation.
4. **Artifact Archival**: Preserving diagnostic reports, confusion matrix heatmaps, and serialized models as downloadable CI build artifacts.

---

### 3. CI/CD ARCHITECTURE
```mermaid
flowchart LR
    GitPush["Git Push / PR"] --> Trigger["GitHub Actions Runner"]
    Trigger --> EnvSetup["Setup Python (3.10, 3.11)"]
    EnvSetup --> Deps["Install Requirements"]
    Deps --> TestUnit["Run Pytest Suite"]
    TestUnit --> ExecPipeline["Run MLOps Pipeline"]
    ExecPipeline --> UploadArtifacts["Upload Reports & Models"]
```

---

### 4. WORKFLOW SPECIFICATION
`.github/workflows/lab8_ci.yml`:
- **Triggers**: Pushes to `main`, `develop`, and `feature/**` branches; pull requests targeting `main`.
- **Matrix Strategy**: OS `ubuntu-latest`, Python runtimes `3.10` and `3.11`.
- **Steps**:
  1. `actions/checkout@v3`: Clones the repository.
  2. `actions/setup-python@v4`: Configures Python with pip caching.
  3. `pip install -r requirements.txt pytest`: Installs project dependencies.
  4. `pytest tests/test_pipeline.py -v`: Executes core pipeline unit tests.
  5. `pytest tests/test_api.py tests/test_performance.py -v`: Tests FastAPI serving endpoints and benchmarks.
  6. `python src/mlops_pipeline/amazon_genre_pipeline.py --force`: Runs full 11-stage MLOps pipeline.
  7. `actions/upload-artifact@v3`: Archives `reports/`, `artifacts/`, and `outputs/`.

---

### 5. CODE FILES & SCRIPTS
- `.github/workflows/lab8_ci.yml`: Master CI pipeline workflow.
- `.github/workflows/lab9_failure_tests.yml`: Scheduled weekly resilience workflow.
- `tests/test_pipeline.py`: Comprehensive test suite for data, model, and reproducibility.
- `tests/test_api.py`: 23 unit tests for FastAPI inference endpoints.
- `tests/test_performance.py`: 6 performance latency and throughput benchmarks.

---

### 6. EXECUTION & VALIDATION COMMANDS
```bash
# Local pre-commit verification command
pytest tests/ -v
python src/mlops_pipeline/amazon_genre_pipeline.py
```

---

### 7. ACTUAL OUTPUTS & VERIFICATION
- Workflow definitions validated with valid YAML syntax.
- All 35 tests verified locally prior to push.
- Matrix configuration verified to support Python 3.10 and 3.11 environments.
- Artifact upload configuration verified to target `reports/`, `artifacts/`, and `outputs/`.

---

### 8. RESULT & CONCLUSION
An automated Continuous Integration (CI) pipeline was designed, configured, and verified using GitHub Actions. The workflow validates the Amazon MLOps codebase across multiple Python environments, automates regression testing, and archives mission-critical reports.

