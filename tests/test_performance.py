import time
import numpy as np
import pytest
from tests.direct_client import DirectTestClient

@pytest.fixture
def client():
    return DirectTestClient()

def test_single_request_latency(client):
    payload = {
        "User Rating": 4.7,
        "Reviews": 17350,
        "Price": 8.0,
        "Year": 2016
    }
    t0 = time.time()
    res = client.post("/predict", json=payload)
    elapsed_ms = (time.time() - t0) * 1000
    assert res.status_code == 200
    assert elapsed_ms < 300.0, f"Latency {elapsed_ms}ms exceeded 300ms threshold"

def test_repeated_inference_average_latency(client):
    payload = {
        "User Rating": 4.5,
        "Reviews": 10000,
        "Price": 12.0,
        "Year": 2017
    }
    latencies = []
    for _ in range(25):
        t0 = time.time()
        res = client.post("/predict", json=payload)
        latencies.append((time.time() - t0) * 1000)
        assert res.status_code == 200

    avg_ms = float(np.mean(latencies))
    p95_ms = float(np.percentile(latencies, 95))
    assert avg_ms < 200.0, f"Average latency {avg_ms}ms exceeded 200ms"
    assert p95_ms < 300.0, f"P95 latency {p95_ms}ms exceeded 300ms"

def test_batch_request_latency(client):
    items = [
        {"User Rating": 4.0 + (i % 10) * 0.1, "Reviews": 1000 * (i + 1), "Price": 5.0 + i, "Year": 2010 + (i % 10)}
        for i in range(25)
    ]
    t0 = time.time()
    res = client.post("/predict/batch", json={"books": items})
    elapsed_ms = (time.time() - t0) * 1000
    assert res.status_code == 200
    assert elapsed_ms < 200.0, f"Batch latency {elapsed_ms}ms exceeded 200ms"

def test_prediction_consistency(client):
    payload = {
        "User Rating": 4.8,
        "Reviews": 35000,
        "Price": 14.0,
        "Year": 2015
    }
    first_res = client.post("/predict", json=payload).json()
    for _ in range(20):
        res = client.post("/predict", json=payload).json()
        assert res["prediction"] == first_res["prediction"]
        assert res["probability_non_fiction"] == first_res["probability_non_fiction"]

def test_error_handling_performance(client):
    payload = {"User Rating": 9.9, "Reviews": 10, "Price": 5.0, "Year": 2015}
    t0 = time.time()
    res = client.post("/predict", json=payload)
    elapsed_ms = (time.time() - t0) * 1000
    assert res.status_code == 422
    assert elapsed_ms < 50.0, f"Error rejection time {elapsed_ms}ms exceeded 50ms"

def test_throughput_stability(client):
    payload = {
        "User Rating": 4.3,
        "Reviews": 5000,
        "Price": 9.0,
        "Year": 2012
    }
    successes = 0
    t0 = time.time()
    iterations = 30
    for _ in range(iterations):
        res = client.post("/predict", json=payload)
        if res.status_code == 200:
            successes += 1
    total_time = time.time() - t0
    qps = iterations / total_time
    assert successes == iterations
    assert qps > 3.0, f"Throughput {qps:.1f} req/s was below 3.0 req/s"
