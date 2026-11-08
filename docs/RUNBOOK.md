# Amazon Bestseller MLOps — Operational Runbook

---

## 1. Quick Start

```bash
# Clone & change to project directory
cd C:\Users\rishi\Desktop\mlopssprooo\amazooonnnn

# Install dependencies
pip install -r requirements.txt

# Run complete MLOps pipeline (all 11 stages)
python src/mlops_pipeline/amazon_genre_pipeline.py --force

# Run all tests
pytest tests/ -v
```

---

## 2. Component Execution Reference

### Stage-by-Stage Execution
| Stage | Command |
| :--- | :--- |
| Data Loading (Lab 1) | `python src/mlops_pipeline/01_load_data.py` |
| Data Validation (Lab 5) | `python src/mlops_pipeline/02_validate_data.py` |
| Preprocessing (Lab 1/5) | `python src/mlops_pipeline/03_preprocessing.py` |
| Output Validation | `python src/mlops_pipeline/04_validate_outputs.py` |
| Train & Register (Lab 2/4/6) | `python src/mlops_pipeline/05_train_registry.py` |
| Evaluate (Lab 2) | `python src/mlops_pipeline/06_model_evaluation.py` |
| Quality Gate (Lab 7) | `python src/mlops_pipeline/07_quality_gate.py` |
| Lifecycle (Lab 6/8) | `python src/mlops_pipeline/08_automate_lifecycle.py` |
| Inference (Lab 10) | `python src/mlops_pipeline/09_inference.py` |
| Failure Injection (Lab 9) | `python src/mlops_pipeline/10_failure_injection.py` |
| Deploy Bundle (Lab 11) | `python src/mlops_pipeline/11_prepare_deployment.py` |

---

## 3. API Operations

```bash
# Start API server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Test health
curl http://localhost:8000/health

# Single book prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"User Rating\": 4.7, \"Reviews\": 17350, \"Price\": 8.0, \"Year\": 2016}"

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d "{\"books\": [{\"User Rating\": 4.7, \"Reviews\": 17350, \"Price\": 8.0, \"Year\": 2016}]}"
```

---

## 4. MLflow UI

```bash
cd C:\Users\rishi\Desktop\mlopssprooo\amazooonnnn
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Visit: http://localhost:5000
```

---

## 5. Docker Operations

```bash
# Build image
docker build -t amazon-genre-api:v1.0 .

# Run container
docker run -d -p 8000:8000 amazon-genre-api:v1.0

# View logs
docker logs amazon-genre-api

# Stop
docker stop amazon-genre-api
```

---

## 6. Kubernetes Operations

```bash
# Apply manifests
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Check status
kubectl get pods -l app=amazon-genre-api
kubectl get service amazon-genre-api-service

# View logs
kubectl logs -l app=amazon-genre-api

# Scale
kubectl scale deployment amazon-genre-api-deployment --replicas=3

# Teardown
kubectl delete -f k8s/
```

---

## 7. Test Execution

```bash
# All tests
pytest tests/ -v

# Specific suites
pytest tests/test_api.py -v              # 23 API tests
pytest tests/test_performance.py -v     # 6 performance tests
pytest tests/test_pipeline.py -v        # 6 pipeline tests
```

---

## 8. Emergency Recovery

If raw data or models are accidentally corrupted, restore from backup:
```bash
copy ".lab9_backup\amazon_bestsellers.csv" "data\raw\amazon_bestsellers.csv"
copy ".lab9_backup\production_model.pkl" "models\production_model.pkl"
```

