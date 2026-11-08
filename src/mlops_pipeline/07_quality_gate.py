import os
import sys
import json
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def run_quality_gate():
    print("=" * 60)
    print("STAGE 7: AUTOMATED QUALITY GATE EVALUATION")
    print("=" * 60)
    
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    eval_report_path = PROJECT_ROOT / "reports" / "evaluation_report.json"
    if not eval_report_path.exists():
        raise FileNotFoundError(f"Evaluation report not found: {eval_report_path}")
        
    with open(eval_report_path, "r", encoding="utf-8") as f:
        eval_report = json.load(f)
        
    gate_cfg = config.get("quality_gate", {})
    min_f1 = float(gate_cfg.get("min_f1_score", 0.70))
    min_accuracy = float(gate_cfg.get("min_accuracy", 0.65))
    
    actual_f1 = float(eval_report.get("f1_weighted", 0.0))
    actual_acc = float(eval_report.get("accuracy", 0.0))
    
    f1_passed = actual_f1 >= min_f1
    acc_passed = actual_acc >= min_accuracy
    overall_status = "PASSED" if (f1_passed and acc_passed) else "FAILED"
    
    gate_decision = {
        "status": overall_status,
        "criteria": {
            "min_f1_weighted": min_f1,
            "min_accuracy": min_accuracy
        },
        "actual_metrics": {
            "f1_weighted": actual_f1,
            "accuracy": actual_acc
        },
        "checks": {
            "f1_gate": "PASSED" if f1_passed else "FAILED",
            "accuracy_gate": "PASSED" if acc_passed else "FAILED"
        },
        "deployment_ready": bool(overall_status == "PASSED")
    }
    
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_file = reports_dir / "quality_gate_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(gate_decision, f, indent=4)
        
    print(f"Quality Gate Status: {overall_status}")
    print(f"F1 Weighted: {actual_f1:.4f} (Threshold: {min_f1}) -> {'PASS' if f1_passed else 'FAIL'}")
    print(f"Accuracy:    {actual_acc:.4f} (Threshold: {min_accuracy}) -> {'PASS' if acc_passed else 'FAIL'}")
    print(f"Report saved to: {out_file}")
    
    if overall_status != "PASSED":
        print("[ERROR] Model failed quality gate. Pipeline halted.")
        sys.exit(1)
        
    return gate_decision

if __name__ == "__main__":
    run_quality_gate()

