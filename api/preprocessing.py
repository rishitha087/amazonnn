import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union

FEATURE_NAMES = ["User Rating", "Reviews_Log", "Price", "Year"]

def preprocess_book_features(data: Union[Dict[str, Any], List[Dict[str, Any]]], scaler) -> np.ndarray:
    """Preprocesses one or more Amazon book items into a scaled feature array.
    
    Transforms 'Reviews' via log1p, extracts User Rating, Reviews_Log, Price, Year,
    and applies standard scaling.
    """
    if isinstance(data, dict):
        items = [data]
    else:
        items = data
        
    records = []
    for item in items:
        # User Rating
        rating = float(item["User Rating"] if "User Rating" in item else item.get("user_rating", 0.0))
        # Reviews
        reviews = float(item["Reviews"] if "Reviews" in item else item.get("reviews", 0.0))
        reviews_log = float(np.log1p(max(0.0, reviews)))
        # Price
        price = float(item["Price"] if "Price" in item else item.get("price", 0.0))
        # Year
        year = float(item["Year"] if "Year" in item else item.get("year", 2019))
        
        records.append([rating, reviews_log, price, year])
        
    arr = np.array(records, dtype=np.float64)
    if scaler is not None:
        df_feat = pd.DataFrame(arr, columns=FEATURE_NAMES)
        scaled = scaler.transform(df_feat)
        return scaled
    return arr
