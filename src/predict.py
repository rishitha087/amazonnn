import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union
import joblib
import numpy as np
import pandas as pd

try:
    from src.config import (
        PRODUCTION_MODEL_PATH, RANDOM_FOREST_MODEL_PATH,
        BASELINE_MODEL_PATH, NOTEBOOK_SAVED_MODEL_PATH,
        PREPROCESSOR_PATH, SCALER_PATH, ENCODER_PATH,
        RAW_DATA_PATH, ALT_RAW_DATA_PATH, PREDICTIONS_DIR,
        RAW_FEATURE_COLUMNS, ENGINEERED_FEATURE_COLUMNS,
        TEST_SIZE, RANDOM_STATE, TARGET_COLUMN
    )
    from src.utils import get_logger, save_json
    logger = get_logger("Predict_Pipeline")
except ImportError:
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODELS_DIR = BASE_DIR / "models"
    OUTPUTS_DIR = BASE_DIR / "outputs"
    PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
    PRODUCTION_MODEL_PATH = MODELS_DIR / "production_model.pkl"
    RANDOM_FOREST_MODEL_PATH = MODELS_DIR / "random_forest_model.pkl"
    BASELINE_MODEL_PATH = MODELS_DIR / "baseline_model.pkl"
    NOTEBOOK_SAVED_MODEL_PATH = BASE_DIR / "bestseller_genre_model.pkl"
    PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
    SCALER_PATH = MODELS_DIR / "scaler.pkl"
    ENCODER_PATH = MODELS_DIR / "encoder.pkl"
    RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "amazon_bestsellers.csv"
    ALT_RAW_DATA_PATH = BASE_DIR / "bestsellers with categories.csv"
    RAW_FEATURE_COLUMNS = ["User Rating", "Reviews", "Price", "Year"]
    ENGINEERED_FEATURE_COLUMNS = ["User Rating", "Reviews_Log", "Price", "Year"]
    TEST_SIZE = 0.20
    RANDOM_STATE = 42
    TARGET_COLUMN = "Genre"
    import logging
    logger = logging.getLogger("Predict_Pipeline")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    def save_json(d, p):
        with open(str(p), "w") as f:
            json.dump(d, f, indent=4)


class AmazonBestsellerPredictor:
    """Production inference engine for Amazon Bestselling Books Genre Classification."""

    def __init__(self, model_path: str = None, preprocessor_path: str = None, encoder_path: str = None):
        # 1. Locate and load model
        if model_path is None:
            for candidate in [PRODUCTION_MODEL_PATH, RANDOM_FOREST_MODEL_PATH, NOTEBOOK_SAVED_MODEL_PATH, BASELINE_MODEL_PATH]:
                if os.path.exists(str(candidate)):
                    model_path = str(candidate)
                    break

        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError("No trained model artifact found in models/ directory.")

        self.model = joblib.load(model_path)
        logger.info(f"Loaded trained model from: {model_path}")

        # 2. Locate and load preprocessor
        self.use_column_transformer = False
        if preprocessor_path and os.path.exists(preprocessor_path):
            self.preprocessor = joblib.load(preprocessor_path)
            self.use_column_transformer = True
            logger.info(f"Loaded ColumnTransformer preprocessor from: {preprocessor_path}")
        elif os.path.exists(str(PREPROCESSOR_PATH)):
            self.preprocessor = joblib.load(str(PREPROCESSOR_PATH))
            self.use_column_transformer = True
            logger.info(f"Loaded ColumnTransformer preprocessor from: {PREPROCESSOR_PATH}")
        elif os.path.exists(str(SCALER_PATH)):
            self.scaler = joblib.load(str(SCALER_PATH))
            self.use_column_transformer = False
            logger.info(f"Loaded separate StandardScaler from: {SCALER_PATH}")
        else:
            raise FileNotFoundError("Preprocessing artifact (preprocessor.pkl or scaler.pkl) not found.")

        # 3. Locate and load LabelEncoder
        if encoder_path and os.path.exists(encoder_path):
            self.encoder = joblib.load(encoder_path)
        elif os.path.exists(str(ENCODER_PATH)):
            self.encoder = joblib.load(str(ENCODER_PATH))
        else:
            self.encoder = None

    def _prepare_features(self, data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]) -> np.ndarray:
        """Transforms raw input dictionary or DataFrame into scaled features array."""
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError("Input data must be a dict, list of dicts, or pandas DataFrame.")

        # Drop non-feature columns if present
        for col in ["Name", "Author", "Genre"]:
            if col in df.columns:
                df = df.drop(col, axis=1)

        # Preprocess features
        if self.use_column_transformer:
            # ColumnTransformer expects RAW_FEATURE_COLUMNS: ["User Rating", "Reviews", "Price", "Year"]
            if "Reviews" not in df.columns and "Reviews_Log" in df.columns:
                # Invert log1p back to reviews if only Reviews_Log was passed
                df["Reviews"] = np.expm1(df["Reviews_Log"])
            return self.preprocessor.transform(df[RAW_FEATURE_COLUMNS])
        else:
            # Fallback to manual log1p + StandardScaler
            if "Reviews_Log" not in df.columns:
                df["Reviews_Log"] = np.log1p(pd.to_numeric(df["Reviews"], errors="coerce").fillna(0))
            features = df[ENGINEERED_FEATURE_COLUMNS]
            return self.scaler.transform(features)

    def predict(self, data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]) -> List[Dict[str, Any]]:
        """Generates predictions and class probabilities for input records."""
        X_features = self._prepare_features(data)
        preds = self.model.predict(X_features)
        probas = self.model.predict_proba(X_features) if hasattr(self.model, "predict_proba") else None

        results = []
        for i, pred in enumerate(preds):
            pred_int = int(pred)
            if self.encoder is not None:
                genre_label = self.encoder.classes_[pred_int]
            else:
                genre_label = "Non Fiction" if pred_int == 1 else "Fiction"

            prob_non_fiction = round(float(probas[i][1]), 4) if probas is not None else None
            confidence = round(float(np.max(probas[i])), 4) if probas is not None else None

            results.append({
                "record_index": i,
                "predicted_class_id": pred_int,
                "predicted_genre": genre_label,
                "confidence": confidence,
                "probability_non_fiction": prob_non_fiction
            })

        return results

    def predict_genre(self, user_rating: float, reviews: float, price: float, year: int) -> str:
        """Matches original Amazon notebook Cell 123 convenience function:
        def predict_genre(user_rating, reviews, price, year): ...
        """
        sample = {
            "User Rating": user_rating,
            "Reviews": reviews,
            "Price": price,
            "Year": year
        }
        res = self.predict(sample)
        return res[0]["predicted_genre"]


def run_prediction_demo():
    """Demonstrates live model loading, sample inference, and unseen test sample prediction."""
    logger.info("Initializing Amazon Bestseller Production Predictor...")
    predictor = AmazonBestsellerPredictor()

    # Demo 1: Notebook custom sample (from Cell 121 & Cell 123)
    sample_book = {
        "User Rating": 4.7,
        "Reviews": 15000,
        "Price": 15,
        "Year": 2020
    }
    pred_res = predictor.predict(sample_book)

    print("\n" + "=" * 65)
    print("AMAZON BESTSELLER GENRE INFERENCE - SPECIFIED SAMPLE")
    print("=" * 65)
    print(f"Book Features: Rating={sample_book['User Rating']} | Reviews={sample_book['Reviews']} | Price=${sample_book['Price']} | Year={sample_book['Year']}")
    print(f"Predicted Genre : {pred_res[0]['predicted_genre']}")
    print(f"Confidence Score: {pred_res[0]['confidence'] * 100:.2f}%")
    print("=" * 65 + "\n")

    # Demo 2: Notebook test set sample verification (matching Cell 129 & 133)
    data_path = RAW_DATA_PATH if RAW_DATA_PATH.exists() else ALT_RAW_DATA_PATH
    if data_path.exists():
        df = pd.read_csv(str(data_path))
        from sklearn.model_selection import train_test_split
        _, test_df = train_test_split(
            df,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=df[TARGET_COLUMN]
        )
        sample_row = test_df.sample(n=1, random_state=RANDOM_STATE).iloc[0]

        actual_genre = sample_row["Genre"]
        input_data = {
            "User Rating": sample_row["User Rating"],
            "Reviews": sample_row["Reviews"],
            "Price": sample_row["Price"],
            "Year": sample_row["Year"]
        }
        prediction = predictor.predict(input_data)[0]

        print("=" * 65)
        print("AMAZON BESTSELLER GENRE INFERENCE - UNSEEN TEST SAMPLE")
        print("=" * 65)
        safe_name = str(sample_row['Name']).encode('ascii', 'replace').decode()
        safe_author = str(sample_row['Author']).encode('ascii', 'replace').decode()
        print(f"Book Title      : '{safe_name}'")
        print(f"Author          : {safe_author}")
        print(f"Actual Genre    : {actual_genre}")
        print(f"Predicted Genre : {prediction['predicted_genre']}")
        print(f"Confidence      : {prediction['confidence'] * 100:.2f}%")
        match = "MATCH (SUCCESS)" if actual_genre == prediction['predicted_genre'] else "MISMATCH"
        print(f"Verification    : {match}")
        print("=" * 65 + "\n")

        # Save sample prediction output
        os.makedirs(str(PREDICTIONS_DIR), exist_ok=True)
        demo_output = {
            "specified_sample": {
                "input": sample_book,
                "prediction": pred_res[0]
            },
            "unseen_test_sample": {
                "title": sample_row["Name"],
                "author": sample_row["Author"],
                "actual_genre": actual_genre,
                "predicted_genre": prediction["predicted_genre"],
                "confidence": prediction["confidence"]
            }
        }
        output_path = PREDICTIONS_DIR / "sample_predictions.json"
        save_json(demo_output, output_path)
        logger.info(f"Saved sample prediction demo output to: {output_path}")

    return pred_res[0]["predicted_genre"]

if __name__ == "__main__":
    run_prediction_demo()
