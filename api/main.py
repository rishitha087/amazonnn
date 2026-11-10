import time
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, ConfigDict

from api.model_loader import get_model, get_scaler, get_metadata
from api.preprocessing import preprocess_book_features

app = FastAPI(
    title="Amazon Bestseller Genre Prediction API",
    description="Production REST API for classifying bestselling books into Fiction or Non Fiction.",
    version="1.0.0"
)

CLASS_MAP = {0: "Fiction", 1: "Non Fiction"}

class BookInput(BaseModel):
    name: Optional[str] = Field(default=None, alias="Name", description="Book title")
    author: Optional[str] = Field(default=None, alias="Author", description="Book author")
    user_rating: float = Field(..., alias="User Rating", ge=0.0, le=5.0, description="User Rating from 0.0 to 5.0")
    reviews: float = Field(..., alias="Reviews", ge=0.0, description="Total number of reviews")
    price: float = Field(..., alias="Price", ge=0.0, description="Price in USD")
    year: int = Field(..., alias="Year", ge=2000, le=2050, description="Publication or bestseller year")

    model_config = ConfigDict(
        populate_by_name=True,
        extra="ignore",
        json_schema_extra={
            "example": {
                "Name": "10-Day Green Smoothie Cleanse",
                "Author": "JJ Smith",
                "User Rating": 4.7,
                "Reviews": 17350,
                "Price": 8.0,
                "Year": 2016
            }
        }
    )

class BatchInput(BaseModel):
    books: List[BookInput]

class PredictionOutput(BaseModel):
    prediction: str
    class_index: int
    probability_non_fiction: float
    probability_fiction: float
    latency_ms: float

class BatchOutput(BaseModel):
    predictions: List[Dict[str, Any]]
    total_samples: int
    latency_ms: float

@app.get("/", tags=["Health"])
def root():
    return {
        "service": "Amazon Bestseller Genre Prediction API",
        "version": "1.0.0",
        "status": "online",
        "docs_url": "/docs"
    }

@app.get("/health", tags=["Health"])
def health_check():
    model_loaded = False
    try:
        model = get_model()
        scaler = get_scaler()
        model_loaded = (model is not None and scaler is not None)
    except Exception:
        model_loaded = False
        
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "service": "Amazon-MLOps-API",
        "timestamp": time.time()
    }

@app.get("/metadata", tags=["Metadata"])
def metadata():
    return get_metadata()

@app.post("/predict", response_model=PredictionOutput, tags=["Inference"])
def predict_single(book: BookInput):
    t0 = time.time()
    try:
        model = get_model()
        scaler = get_scaler()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Model not loaded: {e}")
        
    book_dict = {
        "User Rating": book.user_rating,
        "Reviews": book.reviews,
        "Price": book.price,
        "Year": book.year
    }
    
    scaled_features = preprocess_book_features(book_dict, scaler)
    pred = int(model.predict(scaled_features)[0])
    
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(scaled_features)[0]
        prob_fiction = float(probs[0])
        prob_non_fiction = float(probs[1])
    else:
        prob_non_fiction = 1.0 if pred == 1 else 0.0
        prob_fiction = 1.0 - prob_non_fiction
        
    latency = round((time.time() - t0) * 1000, 2)
    
    return PredictionOutput(
        prediction=CLASS_MAP[pred],
        class_index=pred,
        probability_non_fiction=round(prob_non_fiction, 4),
        probability_fiction=round(prob_fiction, 4),
        latency_ms=latency
    )

@app.post("/predict/batch", response_model=BatchOutput, tags=["Inference"])
def predict_batch(payload: BatchInput):
    t0 = time.time()
    if not payload.books:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Batch list cannot be empty.")
        
    try:
        model = get_model()
        scaler = get_scaler()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Model not loaded: {e}")
        
    items = []
    for b in payload.books:
        items.append({
            "User Rating": b.user_rating,
            "Reviews": b.reviews,
            "Price": b.price,
            "Year": b.year
        })
        
    scaled_features = preprocess_book_features(items, scaler)
    preds = model.predict(scaled_features)
    
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(scaled_features)
    else:
        probs = None
        
    results = []
    for i, p in enumerate(preds):
        p_int = int(p)
        prob_nf = float(probs[i][1]) if probs is not None else (1.0 if p_int == 1 else 0.0)
        results.append({
            "index": i,
            "prediction": CLASS_MAP[p_int],
            "class_index": p_int,
            "probability_non_fiction": round(prob_nf, 4)
        })
        
    latency = round((time.time() - t0) * 1000, 2)
    return BatchOutput(
        predictions=results,
        total_samples=len(results),
        latency_ms=latency
    )

