@echo off
echo ===================================
echo Creating Python Virtual Environment
echo ===================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install wheel for better package handling
echo Installing wheel...
pip install wheel

REM Install requirements
echo Installing requirements from requirements.txt...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install requirements
    echo Trying to install accelerate separately...
    pip install accelerate>=0.20.0
    echo Retrying requirements installation...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install requirements after retry
        pause
        exit /b 1
    )
)

REM Verify critical dependencies
echo Verifying critical dependencies...
python -c "import torch; print('✓ PyTorch installed')" || echo "✗ PyTorch installation failed"
python -c "import transformers; print('✓ Transformers installed')" || echo "✗ Transformers installation failed"
python -c "import accelerate; print('✓ Accelerate installed')" || echo "✗ Accelerate installation failed"
python -c "import streamlit; print('✓ Streamlit installed')" || echo "✗ Streamlit installation failed"

echo.
echo ===================================
echo Environment setup completed successfully!
echo ===================================
echo.
echo To activate the environment in the future, run:
echo   venv\Scripts\activate.bat
echo.
echo To deactivate the environment, run:
echo   deactivate
echo.
echo The environment is now activated and ready to use.
echo.
pause
