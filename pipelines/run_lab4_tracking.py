import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def execute_pipeline():
    print("=" * 70)
    print("STARTING LAB 4: AMAZON MLFLOW EXPERIMENT TRACKING & REPRODUCIBILITY")
    print("=" * 70)

    scripts = [
        "src/preprocess.py",
        "src/train_mlflow.py",
        "src/validate_reproducibility.py"
    ]

    env = os.environ.copy()
    env["MLFLOW_ALLOW_FILE_STORE"] = "true"
    env["MLFLOW_DISABLE_AGENT_HINT"] = "1"

    for rel_path in scripts:
        script_full = BASE_DIR / rel_path
        print(f"\n[INFO] ---> Executing {rel_path}...")
        result = subprocess.run([sys.executable, str(script_full)], cwd=str(BASE_DIR), env=env)

        if result.returncode != 0:
            print(f"\n[ERROR] Pipeline halted. {rel_path} failed with exit code {result.returncode}.")
            sys.exit(1)

    print("\n" + "=" * 70)
    print("[SUCCESS] Lab 4 Amazon MLflow Tracking Pipeline executed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    execute_pipeline()
