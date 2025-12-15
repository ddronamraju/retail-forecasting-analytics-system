"""
Script to execute all analysis notebooks and generate artifacts.

This script runs each module's notebook in the correct order to ensure
all dependencies are met and artifacts are generated.
"""

import os
import subprocess
import sys
from pathlib import Path

# Notebook execution order (some depend on outputs from others)
NOTEBOOKS = [
    "modules/baseline_prophet_forecast/baseline_prophet_forecast.ipynb",
    "modules/feature_engineering/feature_engineering.ipynb",
    "modules/anomaly_detection/anomaly_detection.ipynb",
    "modules/price_elasticity/price_elasticity.ipynb",
    "modules/sku_segmentation/sku_segmentation.ipynb",
    "modules/lgbm_forecasting/lgbm_forecast.ipynb",
    "modules/inventory_risk/inventory_risk.ipynb",
]

def run_notebook(notebook_path):
    """Execute a Jupyter notebook using nbconvert."""
    print(f"\n{'='*60}")
    print(f"Executing: {notebook_path}")
    print(f"{'='*60}")
    
    cmd = [
        "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        "--ExecutePreprocessor.timeout=600",
        notebook_path
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Successfully executed: {notebook_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error executing: {notebook_path}")
        print(f"Error: {e.stderr}")
        return False

def check_artifacts():
    """Check which artifacts were created."""
    artifacts_dir = Path("artifacts")
    if not artifacts_dir.exists():
        print("\n⚠ artifacts/ directory does not exist")
        return
    
    expected_artifacts = [
        "elasticity_store1_dept1.json",
        "sku_clusters.csv",
        "inventory_risk_store1_dept1.csv",
        "inventory_risk_table_store1_dept1.csv",
        "lgbm_model_store1_dept1.pkl",
        "anomalies_store1_dept1.csv"
    ]
    
    print("\n" + "="*60)
    print("Artifact Generation Summary")
    print("="*60)
    
    for artifact in expected_artifacts:
        artifact_path = artifacts_dir / artifact
        if artifact_path.exists():
            size = artifact_path.stat().st_size
            print(f"✓ {artifact} ({size:,} bytes)")
        else:
            print(f"✗ {artifact} (missing)")

def main():
    """Main execution function."""
    print("Retail Forecasting System - Notebook Execution Pipeline")
    print("="*60)
    
    # Check if running from correct directory
    if not Path("modules").exists():
        print("Error: Please run this script from the project root directory")
        sys.exit(1)
    
    # Ensure artifacts directory exists
    Path("artifacts").mkdir(exist_ok=True)
    
    # Track execution status
    success_count = 0
    failed_notebooks = []
    
    # Execute each notebook
    for notebook in NOTEBOOKS:
        notebook_path = Path(notebook)
        
        if not notebook_path.exists():
            print(f"⚠ Skipping missing notebook: {notebook}")
            continue
        
        success = run_notebook(notebook)
        
        if success:
            success_count += 1
        else:
            failed_notebooks.append(notebook)
    
    # Summary
    print("\n" + "="*60)
    print("Execution Summary")
    print("="*60)
    print(f"Total notebooks: {len(NOTEBOOKS)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_notebooks)}")
    
    if failed_notebooks:
        print("\nFailed notebooks:")
        for nb in failed_notebooks:
            print(f"  - {nb}")
    
    # Check artifacts
    check_artifacts()
    
    # Exit code
    if failed_notebooks:
        print("\n⚠ Some notebooks failed to execute")
        sys.exit(1)
    else:
        print("\n✓ All notebooks executed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()
