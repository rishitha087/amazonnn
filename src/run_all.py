"""
Amazon Bestseller Genre Classification - Comprehensive 'src' Pipeline Orchestrator & Automation Engine

This script automates the sequential execution of every executable pipeline component inside `src/`.
Features:
  - Ordered dependency execution
  - Execution time profiling per stage
  - Real-time output streaming
  - Fault tolerance / failure tracking
  - Automated execution report generation in JSON
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone

# Ensure project root is in sys.path
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Complete list of executable scripts in src/ in dependency order
SRC_PIPELINE_STAGES = [
    {
        "id": "step_01_validate_data",
        "name": "Data Quality & Schema Validation",
        "script": "validate_data.py",
        "description": "Validates raw CSV schema and integrity checks"
    },
    {
        "id": "step_02_preprocess",
        "name": "Feature Engineering & Data Preprocessing",
        "script": "preprocess.py",
        "description": "Performs cleaning, feature engineering, and train/test splits"
    },
    {
        "id": "step_03_preprocess_pipeline",
        "name": "Production Preprocessor Pipeline",
        "script": "preprocess_pipeline.py",
        "description": "Fits and serializes Scikit-Learn ColumnTransformer pipeline"
    },
    {
        "id": "step_04_validate_outputs",
        "name": "Preprocessing Output Validation",
        "script": "validate_outputs.py",
        "description": "Validates dimensions, non-emptiness, and sanity of processed arrays"
    },
    {
        "id": "step_05_train_baseline",
        "name": "Baseline Model Training",
        "script": "train.py",
        "description": "Trains initial baseline RandomForestClassifier"
    },
    {
        "id": "step_06_evaluate_baseline",
        "name": "Baseline Model Evaluation",
        "script": "evaluate.py",
        "description": "Evaluates baseline metrics, confusion matrix, error analysis"
    },
    {
        "id": "step_07_train_mlflow",
        "name": "MLflow Experiment Tracking & Grid Search",
        "script": "train_mlflow.py",
        "description": "Tracks 5 algorithms with hyperparameter tuning in MLflow"
    },
    {
        "id": "step_08_validate_reproducibility",
        "name": "Pipeline Determinism & Reproducibility",
        "script": "validate_reproducibility.py",
        "description": "Validates exact seed reproducibility and metric determinism"
    },
    {
        "id": "step_09_train_registry",
        "name": "Model Registry Registration",
        "script": "train_registry.py",
        "description": "Trains candidate model and registers into MLflow Model Registry"
    },
    {
        "id": "step_10_automate_lifecycle",
        "name": "Champion / Challenger Lifecycle Automation",
        "script": "automate_lifecycle.py",
        "description": "Automated promotion of candidate to Staging/Production stage"
    },
    {
        "id": "step_11_generate_registry_report",
        "name": "Production Deployment Governance Report",
        "script": "generate_registry_report.py",
        "description": "Generates deployment report for current production champion"
    },
    {
        "id": "step_12_predict_inference",
        "name": "Production Inference Engine Demo",
        "script": "predict.py",
        "description": "Runs production inference validation on sample and unseen test data"
    }
]

def run_src_pipeline(stop_on_error: bool = True):
    start_total_time = time.time()
    timestamp_str = datetime.now(timezone.utc).isoformat()
    
    print("=" * 80)
    print("   AMAZON BESTSELLER MLOPS - AUTOMATED SRC PIPELINE ORCHESTRATOR")
    print(f"   Execution Timestamp: {timestamp_str}")
    print(f"   Project Root       : {PROJECT_ROOT}")
    print(f"   Total Stages       : {len(SRC_PIPELINE_STAGES)}")
    print("=" * 80)

    stage_results = []
    has_failures = False

    for idx, stage in enumerate(SRC_PIPELINE_STAGES, start=1):
        script_file = SRC_DIR / stage["script"]
        print("\n" + "-" * 80)
        print(f"[{idx}/{len(SRC_PIPELINE_STAGES)}] RUNNING: {stage['name']} ({stage['script']})")
        print(f"Description: {stage['description']}")
        print("-" * 80)

        if not script_file.exists():
            err_msg = f"Script file not found: {script_file}"
            print(f"[ERROR] {err_msg}")
            stage_results.append({
                "stage": stage["id"],
                "name": stage["name"],
                "script": stage["script"],
                "status": "MISSING_FILE",
                "return_code": -1,
                "duration_seconds": 0.0,
                "error": err_msg
            })
            has_failures = True
            if stop_on_error:
                break
            continue

        stage_start = time.time()
        # Run python script using current python interpreter
        cmd = [sys.executable, str(script_file)]
        
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        stage_duration = round(time.time() - stage_start, 2)

        if proc.returncode == 0:
            print(f"\n[PASSED] Stage '{stage['name']}' finished in {stage_duration}s")
            stage_results.append({
                "stage": stage["id"],
                "name": stage["name"],
                "script": stage["script"],
                "status": "PASSED",
                "return_code": 0,
                "duration_seconds": stage_duration
            })
        else:
            print(f"\n[FAILED] Stage '{stage['name']}' failed with code {proc.returncode} after {stage_duration}s")
            stage_results.append({
                "stage": stage["id"],
                "name": stage["name"],
                "script": stage["script"],
                "status": "FAILED",
                "return_code": proc.returncode,
                "duration_seconds": stage_duration
            })
            has_failures = True
            if stop_on_error:
                print(f"\n[HALT] Pipeline halted due to error in stage: {stage['name']}")
                break

    total_duration = round(time.time() - start_total_time, 2)
    overall_status = "FAILED" if has_failures else "SUCCESS"

    # Build and save execution report
    artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    report_file = artifacts_dir / "src_automation_pipeline_report.json"

    report_payload = {
        "pipeline_name": "Amazon Bestsellers Comprehensive 'src' Automation Pipeline",
        "execution_timestamp": timestamp_str,
        "overall_status": overall_status,
        "total_duration_seconds": total_duration,
        "stages_executed": len(stage_results),
        "stages_total": len(SRC_PIPELINE_STAGES),
        "stages": stage_results
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=4)

    # Print summary
    print("\n" + "=" * 80)
    print("   AUTOMATED 'SRC' PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Overall Status   : {overall_status}")
    print(f"Total Duration   : {total_duration}s")
    print(f"Stages Executed  : {len(stage_results)}/{len(SRC_PIPELINE_STAGES)}")
    print(f"Execution Report : {report_file}")
    print("-" * 80)
    for res in stage_results:
        symbol = "PASS" if res["status"] == "PASSED" else "FAIL"
        print(f"  [{symbol:^4}] {res['name']:<45} ({res['duration_seconds']:>6.2f}s) - {res['status']}")
    print("=" * 80)

    if has_failures:
        sys.exit(1)

if __name__ == "__main__":
    stop_on_err = True
    if "--continue-on-error" in sys.argv:
        stop_on_err = False
    run_src_pipeline(stop_on_error=stop_on_err)
