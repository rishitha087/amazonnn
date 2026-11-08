import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def execute_pipeline():
    print("=" * 70)
    print("STARTING LAB 5: AMAZON PRODUCTION DATA PIPELINE & VALIDATION")
    print("=" * 70)

    scripts = [
        "src/validate_data.py",
        "src/preprocess_pipeline.py",
        "src/validate_outputs.py"
    ]

    for rel_path in scripts:
        script_full = BASE_DIR / rel_path
        print(f"\n[INFO] ---> Executing {rel_path}...")
        result = subprocess.run([sys.executable, str(script_full)], cwd=str(BASE_DIR))

        if result.returncode != 0:
            print(f"\n[ERROR] Pipeline halted. {rel_path} failed with exit code {result.returncode}.")
            sys.exit(1)

    print("\n" + "=" * 70)
    print("[SUCCESS] Lab 5 Amazon Production Data Pipeline executed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    execute_pipeline()
