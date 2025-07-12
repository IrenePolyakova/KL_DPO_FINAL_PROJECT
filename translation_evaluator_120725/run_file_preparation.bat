@echo off
echo ===================================
echo  File Preparation App Launcher
echo ===================================

REM Проверка виртуального окружения
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run start.bat and choose option 1 to create environment first.
    pause
    exit /b 1
)

REM Активация виртуального окружения
echo Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Проверка основных библиотек для file_preparation_app.py
echo Checking required libraries for File Preparation App...
echo.

REM Проверка Streamlit
echo Checking Streamlit...
python -c "import streamlit; print(f'✓ Streamlit {streamlit.__version__} - OK')" 2>nul
if %errorlevel% neq 0 (
    echo ✗ Streamlit not found - Installing...
    pip install streamlit>=1.24.0
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install Streamlit
        pause
        exit /b 1
    )
)

REM Проверка Pandas
echo Checking Pandas...
python -c "import pandas; print(f'✓ Pandas {pandas.__version__} - OK')" 2>nul
if %errorlevel% neq 0 (
    echo ✗ Pandas not found - Installing...
    pip install pandas>=1.5.3
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install Pandas
        pause
        exit /b 1
    )
)

REM Проверка openpyxl (для работы с Excel файлами)
echo Checking openpyxl...
python -c "import openpyxl; print(f'✓ openpyxl {openpyxl.__version__} - OK')" 2>nul
if %errorlevel% neq 0 (
    echo ✗ openpyxl not found - Installing...
    pip install openpyxl>=3.0.0
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install openpyxl
        pause
        exit /b 1
    )
)

REM Проверка xlrd (дополнительно для старых Excel файлов)
echo Checking xlrd...
python -c "import xlrd; print(f'✓ xlrd {xlrd.__version__} - OK')" 2>nul
if %errorlevel% neq 0 (
    echo ✗ xlrd not found - Installing...
    pip install xlrd>=2.0.0
    if %errorlevel% neq 0 (
        echo WARNING: Failed to install xlrd (optional dependency)
    )
)

REM Проверка pathlib (обычно встроена в Python 3.4+)
echo Checking pathlib...
python -c "import pathlib; print('✓ pathlib - OK')" 2>nul
if %errorlevel% neq 0 (
    echo ✗ pathlib not found - This should be built-in with Python 3.4+
    echo Please check your Python installation
    pause
    exit /b 1
)

REM Проверка io и zipfile (встроенные модули)
echo Checking built-in modules...
python -c "import io, zipfile, os, logging; print('✓ Built-in modules (io, zipfile, os, logging) - OK')" 2>nul
if %errorlevel% neq 0 (
    echo ✗ Built-in modules not found - Please check your Python installation
    pause
    exit /b 1
)

echo.
echo ===================================
echo All dependencies checked successfully!
echo ===================================
echo.

REM Проверка существования файла приложения
if not exist "file_preparation_app.py" (
    echo ERROR: file_preparation_app.py not found in current directory!
    echo Please make sure the file exists in: %CD%
    pause
    exit /b 1
)

REM Настройка переменных окружения для больших файлов
echo Setting up environment for large file support...
set STREAMLIT_SERVER_MAX_UPLOAD_SIZE=1000
set STREAMLIT_SERVER_MAX_MESSAGE_SIZE=1000
set STREAMLIT_SERVER_ENABLE_CORS=false
set STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
set STREAMLIT_GLOBAL_DEVELOPMENT_MODE=false

REM Создание конфигурационного файла Streamlit
echo Creating Streamlit configuration for File Preparation App...
if not exist ".streamlit" mkdir .streamlit

echo [server] > .streamlit\config.toml
echo maxUploadSize = 1000 >> .streamlit\config.toml
echo maxMessageSize = 1000 >> .streamlit\config.toml
echo enableCORS = false >> .streamlit\config.toml
echo enableXsrfProtection = false >> .streamlit\config.toml
echo headless = false >> .streamlit\config.toml
echo. >> .streamlit\config.toml
echo [browser] >> .streamlit\config.toml
echo gatherUsageStats = false >> .streamlit\config.toml
echo. >> .streamlit\config.toml
echo [global] >> .streamlit\config.toml
echo developmentMode = false >> .streamlit\config.toml

REM Поиск свободного порта для File Preparation App
echo Searching for available port for File Preparation App...
set "AVAILABLE_PORT="

REM Проверяем порты для File Preparation App (отличные от основного приложения)
for %%p in (8511 8512 8513 8514 8515) do (
    netstat -an | findstr ":%%p " >nul 2>&1
    if errorlevel 1 (
        set "AVAILABLE_PORT=%%p"
        goto :found_port
    )
)

REM Если предпочтительные порты заняты, ищем в расширенном диапазоне
for /L %%i in (8516,1,8525) do (
    netstat -an | findstr ":%%i " >nul 2>&1
    if errorlevel 1 (
        set "AVAILABLE_PORT=%%i"
        goto :found_port
    )
)

REM Если все порты заняты, используем автоматический выбор
echo All preferred ports are busy, using automatic port selection...
set "AVAILABLE_PORT=0"

:found_port
if "%AVAILABLE_PORT%"=="0" (
    echo Using automatic port selection
    echo Streamlit will choose an available port automatically
) else (
    echo Using port: %AVAILABLE_PORT%
    echo File Preparation App will be available at: http://localhost:%AVAILABLE_PORT%
)

echo.
echo ===================================
echo Starting File Preparation App
echo ===================================
echo.
echo Configuration:
echo - Max upload size: 1000 MB (1 GB)
echo - Supported formats: CSV, XLSX
echo - Auto-detection of columns
echo - Port: %AVAILABLE_PORT%
echo.
echo Features:
echo - Convert XLSX to CSV UTF-8
echo - Auto-detect source and target columns
echo - Generate multiple file formats
echo - ZIP download with instructions
echo.
echo The application will open in your default browser.
echo Press Ctrl+C to stop the server.
echo.

REM Запуск приложения с выбранным портом
if "%AVAILABLE_PORT%"=="0" (
    streamlit run file_preparation_app.py --server.address=localhost
) else (
    streamlit run file_preparation_app.py --server.port=%AVAILABLE_PORT% --server.address=localhost
)

echo.
echo File Preparation App stopped.
pause
