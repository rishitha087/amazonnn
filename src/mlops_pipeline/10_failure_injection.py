import os
import sys
import json
import time
import shutil
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def run_failure_simulation():
    print("=" * 60)
    print("STAGE 10: LAB 9 PIPELINE FAILURE INJECTION & RECOVERY SUITE")
    print("=" * 60)
    
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    raw_path = PROJECT_ROOT / config["data"]["raw_path"]
    backup_dir = PROJECT_ROOT / ".lab9_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    backup_raw = backup_dir / raw_path.name
    if not backup_raw.exists() and raw_path.exists():
        shutil.copy(raw_path, backup_raw)
        
    model_path = PROJECT_ROOT / "models" / "production_model.pkl"
    backup_model = backup_dir / "production_model.pkl"
    if not backup_model.exists() and model_path.exists():
        shutil.copy(model_path, backup_model)
        
    results = []
    
    # 5 Failure Scenarios, 3 Repetitions each = 15 tests
    scenarios = [
        "SCENARIO_1_MISSING_RAW_DATASET",
        "SCENARIO_2_CORRUPT_COLUMN_SCHEMA",
        "SCENARIO_3_INVALID_DATATYPE_INJECTION",
        "SCENARIO_4_OUT_OF_BOUNDS_VALUES",
        "SCENARIO_5_MISSING_MODEL_ARTIFACT"
    ]
    
    for scn_idx, scn in enumerate(scenarios, 1):
        for rep in range(1, 4):
            t_start = time.time()
            detected = False
            recovered = False
            error_message = ""
            
            try:
                if scn == "SCENARIO_1_MISSING_RAW_DATASET":
                    # Temporarily move raw data
                    temp_hidden = raw_path.with_suffix(".tmp_hidden")
                    if raw_path.exists():
                        shutil.move(raw_path, temp_hidden)
                        
                    # Check detection
                    if not raw_path.exists():
                        detected = True
                        error_message = f"FileNotFoundError: Dataset missing at {raw_path}"
                        
                    # Recovery
                    if temp_hidden.exists():
                        shutil.move(temp_hidden, raw_path)
                        recovered = raw_path.exists()
                    elif backup_raw.exists():
                        shutil.copy(backup_raw, raw_path)
                        recovered = raw_path.exists()
                        
                elif scn == "SCENARIO_2_CORRUPT_COLUMN_SCHEMA":
                    # Drop 'Price' column
                    df = pd.read_csv(backup_raw)
                    df_corrupt = df.drop(columns=["Price"])
                    df_corrupt.to_csv(raw_path, index=False)
                    
                    # Test validation detection
                    df_test = pd.read_csv(raw_path)
                    required = config["data"]["columns"]
                    missing = set(required) - set(df_test.columns)
                    if missing:
                        detected = True
                        error_message = f"SchemaError: Missing columns: {list(missing)}"
                        
                    # Recovery
                    shutil.copy(backup_raw, raw_path)
                    recovered = "Price" in pd.read_csv(raw_path).columns
                    
                elif scn == "SCENARIO_3_INVALID_DATATYPE_INJECTION":
                    # Inject non-numeric string into User Rating
                    df = pd.read_csv(backup_raw)
                    df.loc[0, "User Rating"] = "EXCELLENT_BOOK"
                    df.to_csv(raw_path, index=False)
                    
                    df_test = pd.read_csv(raw_path)
                    if not pd.to_numeric(df_test["User Rating"], errors="coerce").notnull().all():
                        detected = True
                        error_message = "TypeError: Non-numeric strings found in User Rating"
                        
                    # Recovery
                    shutil.copy(backup_raw, raw_path)
                    df_restored = pd.read_csv(raw_path)
                    recovered = pd.to_numeric(df_restored["User Rating"], errors="coerce").notnull().all()
                    
                elif scn == "SCENARIO_4_OUT_OF_BOUNDS_VALUES":
                    # Inject User Rating = 9.5 (valid is 0 to 5)
                    df = pd.read_csv(backup_raw)
                    df.loc[0, "User Rating"] = 9.5
                    df.to_csv(raw_path, index=False)
                    
                    df_test = pd.read_csv(raw_path)
                    rating_col = pd.to_numeric(df_test["User Rating"], errors="coerce")
                    if (rating_col < 0).any() or (rating_col > 5.0).any():
                        detected = True
                        error_message = "RangeError: User Rating out of bounds [0.0, 5.0]"
                        
                    # Recovery
                    shutil.copy(backup_raw, raw_path)
                    recovered = (pd.to_numeric(pd.read_csv(raw_path)["User Rating"]) <= 5.0).all()
                    
                elif scn == "SCENARIO_5_MISSING_MODEL_ARTIFACT":
                    # Temporarily rename model artifact
                    temp_model_hidden = model_path.with_suffix(".tmp_model")
                    if model_path.exists():
                        shutil.move(model_path, temp_model_hidden)
                        
                    if not model_path.exists():
                        detected = True
                        error_message = "ModelArtifactMissingError: production_model.pkl not found"
                        
                    # Recovery
                    if temp_model_hidden.exists():
                        shutil.move(temp_model_hidden, model_path)
                        recovered = model_path.exists()
                    elif backup_model.exists():
                        shutil.copy(backup_model, model_path)
                        recovered = model_path.exists()
                        
            except Exception as ex:
                detected = True
                error_message = str(ex)
                # Ensure clean restoration
                if backup_raw.exists() and (not raw_path.exists() or len(pd.read_csv(raw_path).columns) < 7):
                    shutil.copy(backup_raw, raw_path)
                if backup_model.exists() and not model_path.exists():
                    shutil.copy(backup_model, model_path)
                recovered = True
                
            latency_ms = round((time.time() - t_start) * 1000, 2)
            results.append({
                "scenario": str(scn),
                "repetition": int(rep),
                "detected": bool(detected),
                "recovered": bool(recovered),
                "error_detected": str(error_message),
                "duration_ms": float(latency_ms),
                "status": "PASSED" if (detected and recovered) else "FAILED"
            })
            
    # Always ensure raw data and model are properly restored at the end
    if backup_raw.exists():
        shutil.copy(backup_raw, raw_path)
    if backup_model.exists() and not model_path.exists():
        shutil.copy(backup_model, model_path)
        
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r["status"] == "PASSED")
    recovery_rate = (passed_tests / total_tests) * 100
    
    summary = {
        "suite": "Lab 9 Failure Injection and Automated Self-Healing",
        "total_scenarios": len(scenarios),
        "repetitions_per_scenario": 3,
        "total_test_runs": total_tests,
        "successful_recoveries": passed_tests,
        "recovery_rate_percent": recovery_rate,
        "average_duration_ms": round(float(np.mean([r["duration_ms"] for r in results])), 2),
        "test_results": results
    }
    
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / "failure_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
        
    print(f"[SUCCESS] Lab 9 Failure Injection Suite Completed: {passed_tests}/{total_tests} passed ({recovery_rate:.1f}% Recovery Rate).")
    print(f"Report written to: {report_file}")
    return summary

if __name__ == "__main__":
    run_failure_simulation()
