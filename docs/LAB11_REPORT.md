# LABORATORY EXPERIMENT 11
## Containerization and Deployment Using Docker and Kubernetes

---

### 1. AIM
To containerize the Amazon Bestseller Genre Prediction REST API using Docker, create production-ready container images following security best practices, and design Kubernetes manifests for high-availability, horizontally-scalable production deployment.

---

### 2. THEORY & BACKGROUND
Containerization isolates the runtime environment, eliminating "works on my machine" defects and enabling reproducible deployments across development, staging, and production environments:

1. **Docker Containerization**:
   - Encapsulates the Python runtime, installed dependencies, and model artifacts into a self-contained, immutable image layer.
   - Uses a slim base image (`python:3.11-slim`) to minimize attack surface and reduce image pull times.
   - Non-root user execution (`appuser`) follows the principle of least privilege.
   - Native HEALTHCHECK ensures automated container orchestrators can verify service liveness.
2. **Kubernetes Orchestration**:
   - `Deployment` manifest maintains 2 replica pods behind an internal load balancer.
   - Resource quotas (`cpu: 100m-500m`, `memory: 128Mi-512Mi`) prevent resource contention.
   - `livenessProbe` and `readinessProbe` targeting `GET /health` enable automatic pod replacement and controlled traffic routing.
   - `NodePort` Service exposes the application on external port 30080 for lab environment access.

---

### 3. SOFTWARE & SYSTEM SPECIFICATIONS
- **Base Image**: `python:3.11-slim`
- **Containerization Platform**: Docker Engine
- **Orchestration**: Kubernetes
- **External Port**: NodePort 30080
- **Internal Port**: Container port 8000

---

### 4. IMPLEMENTATION DETAILS
**`Dockerfile`** structure:
1. Copies `requirements-api.txt` and installs with `--no-cache-dir`.
2. Copies `api/`, `deployment/`, and `config.yaml`.
3. Creates unprivileged `appuser` for runtime.
4. Declares `HEALTHCHECK` via `curl /health`.
5. Runs `uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2`.

**`k8s/deployment.yaml`**:
- `replicas: 2` for high-availability.
- Rolling update strategy (default).
- Liveness probe: `GET /health`, delay 10s, period 15s.
- Readiness probe: `GET /health`, delay 5s, period 10s.

**`k8s/service.yaml`**:
- Type `NodePort`, port 8000, nodePort 30080.

---

### 5. CODE FILES & SCRIPTS
- `Dockerfile`: Full production Docker image definition.
- `.dockerignore`: Excludes data, mlruns, notebooks, and dev-only directories from image.
- `k8s/deployment.yaml`: Kubernetes Deployment specification.
- `k8s/service.yaml`: Kubernetes NodePort Service definition.
- `requirements-api.txt`: Minimal API runtime dependencies.

---

### 6. EXECUTION COMMANDS
```bash
# Build Docker Image
docker build -t amazon-genre-api:v1.0 .

# Run locally for testing
docker run -d -p 8000:8000 --name amazon-genre-api amazon-genre-api:v1.0

# Verify health
curl http://localhost:8000/health

# Apply to Kubernetes cluster
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Verify running pods
kubectl get pods -l app=amazon-genre-api
kubectl get service amazon-genre-api-service

# Test prediction via NodePort
curl -X POST http://localhost:30080/predict \
  -H "Content-Type: application/json" \
  -d '{"User Rating": 4.7, "Reviews": 17350, "Price": 8.0, "Year": 2016}'
```

---

### 7. ACTUAL OUTPUTS & VERIFICATION
Container build produces:
- `amazon-genre-api:v1.0` (Base: `python:3.11-slim`, Dependencies installed, Users created)

Kubernetes verification:
- Deployment with 2/2 Ready replicas
- Service LoadBalancer forwarding :30080 → Pod :8000

Health endpoint response:
```json
{
    "status": "healthy",
    "model_loaded": true,
    "service": "Amazon-MLOps-API",
    "timestamp": 1731067500.0
}
```

---

### 8. RESULT & CONCLUSION
The Amazon Bestseller Genre Prediction API was successfully containerized using Docker with security hardening, and Kubernetes manifests were created for high-availability production deployment. The system supports 2-replica pod execution with automated health probing and rolling updates.

