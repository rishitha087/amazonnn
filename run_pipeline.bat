@echo off
REM ==============================================================================
REM Amazon Bestselling Books MLOps - 1-Click Pipeline Automation
REM ==============================================================================

echo [INFO] Starting Automated Amazon MLOps Pipeline in src/ ...
python -m src.run_all

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Pipeline execution failed with exit code %ERRORLEVEL%.
    exit /b %ERRORLEVEL%
)

echo [INFO] Running test suite...
pytest tests/ -v

if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Some tests encountered issues with exit code %ERRORLEVEL%.
    exit /b %ERRORLEVEL%
)

echo ==============================================================================
echo [SUCCESS] Complete Amazon MLOps Pipeline and Tests executed successfully!
echo ==============================================================================
