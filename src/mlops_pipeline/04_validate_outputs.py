import sys
import json
import yaml
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def validate_outputs():
    print("=" * 60)
    print("STAGE 4: VALIDATING PREPROCESSED ARRAYS & OUTPUTS")
    print("=" * 60)
    
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]
    
    x_train_path = processed_dir / "X_train_final.npy"
    x_test_path = processed_dir / "X_test_final.npy"
    y_train_path = processed_dir / "y_train.npy"
    y_test_path = processed_dir / "y_test.npy"
    
    for p in [x_train_path, x_test_path, y_train_path, y_test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing processed array: {p}")
            
    X_train = np.load(x_train_path)
    X_test = np.load(x_test_path)
    y_train = np.load(y_train_path)
    y_test = np.load(y_test_path)
    
    # Check NaN and Inf
    assert not np.isnan(X_train).any(), "NaN found in X_train"
    assert not np.isnan(X_test).any(), "NaN found in X_test"
    assert not np.isinf(X_train).any(), "Inf found in X_train"
    assert not np.isinf(X_test).any(), "Inf found in X_test"
    
    # Check shapes
    expected_features = 4
    assert X_train.shape[1] == expected_features, f"Expected {expected_features} features in X_train, got {X_train.shape[1]}"
    assert X_test.shape[1] == expected_features, f"Expected {expected_features} features in X_test, got {X_test.shape[1]}"
    assert len(X_train) == len(y_train), "X_train and y_train length mismatch"
    assert len(X_test) == len(y_test), "X_test and y_test length mismatch"
    
    # Check binary labels
    unique_train = set(np.unique(y_train))
    unique_test = set(np.unique(y_test))
    assert unique_train.issubset({0, 1}), f"Invalid labels in y_train: {unique_train}"
    assert unique_test.issubset({0, 1}), f"Invalid labels in y_test: {unique_test}"
    
    report = {
        "status": "PASSED",
        "X_train_shape": list(X_train.shape),
        "X_test_shape": list(X_test.shape),
        "y_train_shape": list(y_train.shape),
        "y_test_shape": list(y_test.shape),
        "train_mean": float(np.mean(X_train)),
        "train_std": float(np.std(X_train)),
        "nan_count": 0,
        "inf_count": 0,
        "classes": [int(c) for c in sorted(unique_train)]
    }
    
    artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    report_path = artifacts_dir / "output_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
        
    print(f"[SUCCESS] Processed arrays validated: X_train={X_train.shape}, X_test={X_test.shape}")
    print(f"Validation report saved to: {report_path}")
    return report

if __name__ == "__main__":
    validate_outputs()

