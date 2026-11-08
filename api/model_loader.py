import os
import json
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

_MODEL = None
_SCALER = None
_METADATA = None

def get_model():
    global _MODEL
    if _MODEL is None:
        candidates = [
            BASE_DIR / "deployment" / "production_model.pkl",
            BASE_DIR / "models" / "production_model.pkl",
            BASE_DIR / "models" / "random_forest_model.pkl",
            BASE_DIR / "bestseller_genre_model.pkl"
        ]
        for c in candidates:
            if c.exists():
                _MODEL = joblib.load(c)
                break
        if _MODEL is None:
            raise FileNotFoundError("No production model artifact found.")
    return _MODEL

def get_scaler():
    global _SCALER
    if _SCALER is None:
        candidates = [
            BASE_DIR / "deployment" / "scaler.pkl",
            BASE_DIR / "models" / "scaler.pkl"
        ]
        for c in candidates:
            if c.exists():
                _SCALER = joblib.load(c)
                break
    return _SCALER

def get_metadata():
    global _METADATA
    if _METADATA is None:
        meta_path = BASE_DIR / "deployment" / "model_metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                _METADATA = json.load(f)
        else:
            _METADATA = {
                "model_name": "Amazon_Bestseller_Production_Model",
                "task": "Binary Classification (Fiction / Non Fiction)",
                "classes": {0: "Fiction", 1: "Non Fiction"},
                "features": ["User Rating", "Reviews_Log", "Price", "Year"]
            }
    return _METADATA

