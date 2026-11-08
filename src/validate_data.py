import os
from pathlib import Path
import pandas as pd

try:
    import pandera as pa
    from pandera import Column, Check
except ImportError:
    pa = None

try:
    from src.config import RAW_DATA_PATH, ALT_RAW_DATA_PATH, ARTIFACTS_DIR, TARGET_COLUMN
    from src.utils import get_logger
    logger = get_logger("Validate_Data_Lab5")
except ImportError:
    BASE_DIR = Path(__file__).resolve().parent.parent
    RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "amazon_bestsellers.csv"
    ALT_RAW_DATA_PATH = BASE_DIR / "bestsellers with categories.csv"
    ARTIFACTS_DIR = BASE_DIR / "artifacts"
    TARGET_COLUMN = "Genre"
    import logging
    logger = logging.getLogger("Validate_Data_Lab5")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

def get_amazon_bestseller_schema():
    """Defines strict Pandera schema specification for the Amazon Bestselling Books dataset."""
    if pa is None:
        raise ImportError("Pandera is required for schema validation.")

    return pa.DataFrameSchema({
        "Name": Column(pa.String, nullable=False),
        "Author": Column(pa.String, nullable=False),
        "User Rating": Column(pa.Float, Check.in_range(1.0, 5.0), nullable=False),
        "Reviews": Column(pa.Int, Check.ge(0), nullable=False),
        "Price": Column(pa.Int, Check.ge(0), nullable=False),
        "Year": Column(pa.Int, Check.in_range(2000, 2030), nullable=False),
        "Genre": Column(pa.String, Check.isin(["Fiction", "Non Fiction"]), nullable=False)
    }, strict=False)

def validate_raw_data(data_path=None, output_report_name="schema_validation_errors.csv"):
    """Performs comprehensive validation of the raw Amazon Bestsellers dataset before training."""
    if data_path is None:
        data_path = RAW_DATA_PATH if RAW_DATA_PATH.exists() else ALT_RAW_DATA_PATH

    logger.info(f"Starting raw data validation on: {data_path}...")

    # 1. File existence
    if not os.path.exists(str(data_path)):
        logger.error(f"[ERROR] Raw dataset file not found at: {data_path}")
        return False

    df = pd.read_csv(str(data_path))
    logger.info(f"Loaded raw dataset ({len(df)} records, {len(df.columns)} columns).")

    errors = []

    # 2. Check row count
    if len(df) < 100:
        errors.append(f"Insufficient row count: found {len(df)}, expected at least 100.")

    # 3. Check required columns and target
    expected_cols = ["Name", "Author", "User Rating", "Reviews", "Price", "Year", "Genre"]
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    # 4. Check target variable
    if TARGET_COLUMN in df.columns:
        unexpected_genres = set(df[TARGET_COLUMN].dropna().unique()) - {"Fiction", "Non Fiction"}
        if unexpected_genres:
            errors.append(f"Unexpected target values in '{TARGET_COLUMN}': {unexpected_genres}")
        if df[TARGET_COLUMN].isnull().sum() > 0:
            errors.append(f"Target '{TARGET_COLUMN}' contains {df[TARGET_COLUMN].isnull().sum()} nulls.")

    # 5. Missing values check
    total_nulls = int(df.isnull().sum().sum())
    if total_nulls > 0:
        logger.warning(f"Detected {total_nulls} missing values across raw dataset.")

    # 6. Duplicates check
    exact_duplicates = int(df.duplicated().sum())
    logger.info(f"Exact duplicate rows: {exact_duplicates}")

    # 7. Pandera Schema Validation
    schema_passed = True
    if pa is not None:
        try:
            schema = get_amazon_bestseller_schema()
            schema.validate(df, lazy=True)
            logger.info("[SUCCESS] Pandera Schema Validation PASSED. All columns, types, and ranges are valid.")
        except pa.errors.SchemaErrors as err:
            schema_passed = False
            failures = err.failure_cases[['schema_context', 'column', 'check', 'failure_case', 'index']]
            logger.error(f"[ERROR] Schema Validation FAILED with errors:\n{failures.to_string()}")
            os.makedirs(str(ARTIFACTS_DIR), exist_ok=True)
            report_path = os.path.join(str(ARTIFACTS_DIR), output_report_name)
            failures.to_csv(report_path, index=False)
            logger.info(f"Failure report saved to: {report_path}")
            errors.append("Pandera schema validation failed.")

    if errors:
        logger.error(f"[FAILED] Data Validation FAILED with {len(errors)} issues:")
        for err in errors:
            logger.error(f"  - {err}")
        return False

    logger.info("[SUCCESS] Amazon raw dataset passed all validation checks! Ready for production pipeline.")
    return True

if __name__ == "__main__":
    validate_raw_data()
