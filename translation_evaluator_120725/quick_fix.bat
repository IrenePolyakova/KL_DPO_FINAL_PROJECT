@echo off
echo ===================================
echo  Quick Fix for Model Loading Issues
echo ===================================

REM Проверка виртуального окружения
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run start.bat and choose option 1 first.
    pause
    exit /b 1
)

REM Активация виртуального окружения
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Installing accelerate (required for model loading)...
pip install accelerate>=0.20.0

echo.
echo Checking if all dependencies are properly installed...
python -c "import accelerate; print('✓ Accelerate is now installed')" 2>nul || echo "✗ Accelerate installation failed"
python -c "import torch; print('✓ PyTorch is available')" 2>nul || echo "✗ PyTorch not found"
python -c "import transformers; print('✓ Transformers is available')" 2>nul || echo "✗ Transformers not found"

echo.
echo ===================================
echo Fix completed!
echo ===================================
echo.
echo Your application should now work properly.
echo The model loading error should be resolved.
echo.
echo You can now run the application using:
echo - start.bat (choose option 2 or 3)
echo.
pause
