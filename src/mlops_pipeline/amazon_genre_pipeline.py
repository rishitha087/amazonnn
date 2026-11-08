import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STAGES = [
    ("Stage 01: Raw Data Ingestion & Extraction", "01_load_data.py"),
    ("Stage 02: Schema & Distribution Data Validation", "02_validate_data.py"),
    ("Stage 03: Feature Engineering & Preprocessing", "03_preprocessing.py"),
    ("Stage 04: Processed Arrays & Output Validation", "04_validate_outputs.py"),
    ("Stage 05: Model Training & MLflow Registry", "05_train_registry.py"),
    ("Stage 06: Model Evaluation & Error Analysis", "06_model_evaluation.py"),
    ("Stage 07: Quality Gate & Performance Enforcement", "07_quality_gate.py"),
    ("Stage 08: Model Lifecycle & Stage Transition", "08_automate_lifecycle.py"),
    ("Stage 09: Batch Production Inference", "09_inference.py"),
    ("Stage 10: Failure Injection & Recovery Suite", "10_failure_injection.py"),
    ("Stage 11: Production Deployment Preparation", "11_prepare_deployment.py")
]

def run_pipeline(force: bool = False):
    print("=" * 80)
    print("AMAZON BESTSELLER GENRE PREDICTION - MASTER MLOPS PIPELINE")
    print("=" * 80)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Force Re-run: {force}\n")
    
    pipeline_dir = PROJECT_ROOT / "src" / "mlops_pipeline"
    total_start = time.time()
    stage_results = []
    
    for idx, (title, script_name) in enumerate(STAGES, 1):
        script_path = pipeline_dir / script_name
        print(f"\n[{idx}/{len(STAGES)}] RUNNING: {title} ({script_name})")
        print("-" * 80)
        
        t0 = time.time()
        res = subprocess.run([sys.executable, str(script_path)], cwd=str(PROJECT_ROOT))
        elapsed = round(time.time() - t0, 2)
        
        if res.returncode == 0:
            print(f"--> [PASS] {title} completed in {elapsed}s")
            stage_results.append((title, "PASS", elapsed))
        else:
            print(f"--> [FAIL] {title} failed with code {res.returncode}")
            stage_results.append((title, f"FAIL (Code {res.returncode})", elapsed))
            if not force:
                print("\n[FATAL] Pipeline stopped due to failure. Use --force to proceed regardless.")
                sys.exit(res.returncode)
                
    total_elapsed = round(time.time() - total_start, 2)
    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    for title, status, elapsed in stage_results:
        print(f"  {status.ljust(8)} | {str(elapsed).rjust(6)}s | {title}")
    print("-" * 80)
    print(f"Total Execution Time: {total_elapsed}s")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Amazon Master MLOps Pipeline")
    parser.add_argument("--force", action="store_true", help="Continue even if a stage fails")
    args = parser.parse_args()
    run_pipeline(force=args.force)

