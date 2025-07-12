@echo off
echo ===================================
echo  Testing Dependencies
echo ===================================

REM Проверка виртуального окружения
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup_env.bat first.
    pause
    exit /b 1
)

REM Активация виртуального окружения
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Running dependency test...
python test_dependencies.py

echo.
echo Test completed.
pause
