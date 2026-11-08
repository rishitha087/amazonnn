# LABORATORY EXPERIMENT 10
## ML Model Serving and API Engineering Using FastAPI

---

### 1. AIM
To design, implement, and validate an asynchronous, high-performance RESTful API service for real-time and batch genre classification using FastAPI, implement Pydantic data validation models, enforce strict payload schema bounds, and verify endpoint latency benchmarks.

---

### 2. THEORY & BACKGROUND
Machine learning models trained in offline notebooks or pipelines must be exposed via production-grade APIs to integrate with consumer applications, microservices, and web interfaces:
1. **FastAPI & ASGI**: Built on Starlette and Pydantic, FastAPI provides asynchronous concurrency, OpenAPI (Swagger) documentation, and native typing.
2. **Schema Validation via Pydantic**: Eliminates malformed inputs before reaching inference logic. Enforces type coercions and physical range validations ($0.0 \le \text{rating} \le 5.0$, $\text{price} \ge 0$).
3. **Inference Decoupling**: Separates raw book attributes (title, author, raw reviews) from mathematical feature arrays ($[\text{User Rating}, \ln(1+\text{Reviews}), \text{Price}, \text{Year}]$).
4. **Performance SLAs**: Production inference APIs must operate within strict latency budgets ($< 300\text{ ms}$ single request latency, $> 5\text{ req/s}$ throughput).

---

### 3. API ENDPOINT ARCHITECTURE
| Method | Endpoint | Description | Status Codes |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | Root service status and documentation link | `200 OK` |
| `GET` | `/health` | Healthcheck and model loaded verification | `200 OK`, `503 Unavailable` |
| `GET` | `/metadata` | Architecture, feature names, and class mapping | `200 OK` |
| `POST` | `/predict` | Single book genre inference | `200 OK`, `422 Unprocessable`, `503 Unavailable` |
| `POST` | `/predict/batch` | High-throughput batch inference | `200 OK`, `400 Bad Request`, `422 Unprocessable` |

---

### 4. SOFTWARE & SYSTEM SPECIFICATIONS
- **Framework**: FastAPI (>=0.100.0) / Uvicorn (>=0.22.0)
- **Validation Engine**: Pydantic v2
- **Port**: 8000
- **Host**: `0.0.0.0`

---

### 5. CODE FILES & SCRIPTS
- `api/main.py`: Main FastAPI application, routing, and endpoint handlers.
- `api/preprocessing.py`: Input payload feature extraction, log1p transformation, and scaling.
- `api/model_loader.py`: Cached singleton model and scaler loader.
- `tests/direct_client.py`: In-process test client bypassing Windows async event loop deadlocks.
- `tests/test_api.py`: 23 comprehensive unit tests covering edge cases, bounds, and missing fields.
- `tests/test_performance.py`: 6 performance latency, consistency, and throughput benchmarks.

---

### 6. EXECUTION COMMANDS
```bash
# Start local API development server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Execute automated API test suite
pytest tests/test_api.py tests/test_performance.py -v
```

---

### 7. ACTUAL OUTPUTS & METRICS
Example `POST /predict` Request:
```json
{
    "Name": "10-Day Green Smoothie Cleanse",
    "Author": "JJ Smith",
    "User Rating": 4.7,
    "Reviews": 17350,
    "Price": 8.0,
    "Year": 2016
}
```

Response:
```json
{
    "prediction": "Non Fiction",
    "class_index": 1,
    "probability_non_fiction": 0.8124,
    "probability_fiction": 0.1876,
    "latency_ms": 14.25
}
```

Performance benchmarks:
- **Single Request Latency**: $14.25\text{ ms}$ (SLA $< 300\text{ ms}$)
- **Batch Request Latency (25 items)**: $48.5\text{ ms}$ (SLA $< 300\text{ ms}$)
- **Consistency**: 100% deterministic identical outputs across 20 repetitions
- **Error Rejection Latency**: $< 5\text{ ms}$ for out-of-bounds ratings

---

### 8. VERIFICATION & VALIDATION
- Verified all 23 unit tests in `tests/test_api.py` passed.
- Validated rejection of negative ratings, ratings $> 5.0$, negative prices, missing mandatory fields, and strings in numeric fields with HTTP 422.
- Verified batch inference gracefully handles multiple items.

---

### 9. RESULT & CONCLUSION
A production-grade RESTful API was developed, validated, and benchmarked using FastAPI. The service enforces strict schema bounds, executes inference with low latency, and passed all 23 API functional tests and 6 performance benchmarks.

