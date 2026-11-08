import pytest
from tests.direct_client import DirectTestClient

@pytest.fixture
def client():
    return DirectTestClient()

def test_health_check(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True

def test_metadata_endpoint(client):
    res = client.get("/metadata")
    assert res.status_code == 200
    data = res.json()
    assert "classes" in data
    assert "features" in data or "input_features" in data

def test_predict_single_valid_nonfiction(client):
    payload = {
        "Name": "10-Day Green Smoothie Cleanse",
        "Author": "JJ Smith",
        "User Rating": 4.7,
        "Reviews": 17350,
        "Price": 8.0,
        "Year": 2016
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["prediction"] in ["Fiction", "Non Fiction"]
    assert data["class_index"] in [0, 1]
    assert 0.0 <= data["probability_non_fiction"] <= 1.0

def test_predict_single_valid_fiction(client):
    payload = {
        "Name": "The Alchemist",
        "Author": "Paulo Coelho",
        "User Rating": 4.7,
        "Reviews": 35799,
        "Price": 15.0,
        "Year": 2014
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["prediction"] in ["Fiction", "Non Fiction"]

def test_predict_single_zero_price(client):
    payload = {
        "User Rating": 4.5,
        "Reviews": 1000,
        "Price": 0.0,
        "Year": 2015
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200

def test_predict_single_zero_reviews(client):
    payload = {
        "User Rating": 4.0,
        "Reviews": 0.0,
        "Price": 10.0,
        "Year": 2018
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200

def test_predict_single_max_rating(client):
    payload = {
        "User Rating": 5.0,
        "Reviews": 5000,
        "Price": 20.0,
        "Year": 2019
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200

def test_predict_single_min_rating(client):
    payload = {
        "User Rating": 0.0,
        "Reviews": 100,
        "Price": 5.0,
        "Year": 2010
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200

def test_predict_negative_rating_fails(client):
    payload = {
        "User Rating": -1.0,
        "Reviews": 100,
        "Price": 5.0,
        "Year": 2010
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 422

def test_predict_excessive_rating_fails(client):
    payload = {
        "User Rating": 5.5,
        "Reviews": 100,
        "Price": 5.0,
        "Year": 2010
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 422

def test_predict_negative_reviews_fails(client):
    payload = {
        "User Rating": 4.5,
        "Reviews": -50,
        "Price": 10.0,
        "Year": 2015
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 422

def test_predict_negative_price_fails(client):
    payload = {
        "User Rating": 4.5,
        "Reviews": 500,
        "Price": -10.0,
        "Year": 2015
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 422

def test_predict_year_too_early_fails(client):
    payload = {
        "User Rating": 4.5,
        "Reviews": 500,
        "Price": 10.0,
        "Year": 1990
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 422

def test_predict_missing_rating_fails(client):
    payload = {
        "Reviews": 500,
        "Price": 10.0,
        "Year": 2015
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 422

def test_predict_missing_reviews_fails(client):
    payload = {
        "User Rating": 4.5,
        "Price": 10.0,
        "Year": 2015
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 422

def test_predict_missing_price_fails(client):
    payload = {
        "User Rating": 4.5,
        "Reviews": 500,
        "Year": 2015
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 422

def test_predict_missing_year_fails(client):
    payload = {
        "User Rating": 4.5,
        "Reviews": 500,
        "Price": 10.0
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 422

def test_predict_string_rating_fails(client):
    payload = {
        "User Rating": "excellent",
        "Reviews": 500,
        "Price": 10.0,
        "Year": 2015
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 422

def test_predict_extra_fields_ignored(client):
    payload = {
        "User Rating": 4.8,
        "Reviews": 12000,
        "Price": 12.0,
        "Year": 2017,
        "Publisher": "Penguin Random House",
        "ISBN": "123-4567890123"
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200

def test_predict_probabilities_sum_to_one(client):
    payload = {
        "User Rating": 4.6,
        "Reviews": 8500,
        "Price": 14.0,
        "Year": 2018
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    total = data["probability_non_fiction"] + data["probability_fiction"]
    assert round(total, 2) == 1.0

def test_predict_batch_success(client):
    payload = {
        "books": [
            {"User Rating": 4.7, "Reviews": 17350, "Price": 8.0, "Year": 2016},
            {"User Rating": 4.6, "Reviews": 2052, "Price": 22.0, "Year": 2011},
            {"User Rating": 4.8, "Reviews": 35799, "Price": 15.0, "Year": 2014}
        ]
    }
    res = client.post("/predict/batch", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["total_samples"] == 3
    assert len(data["predictions"]) == 3

def test_predict_batch_empty_fails(client):
    payload = {"books": []}
    res = client.post("/predict/batch", json=payload)
    assert res.status_code in [400, 422]

def test_unknown_route_returns_404(client):
    res = client.get("/unknown_endpoint")
    assert res.status_code == 404

