@echo off
echo ===================================
echo  Translation Evaluator Launcher
echo ===================================

REM Проверка активации виртуального окружения
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup_env.bat first to create the environment.
    pause
    exit /b 1
)

REM Активация виртуального окружения
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Проверка установки зависимостей
echo Checking dependencies...
python -c "import streamlit" 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Streamlit not installed!
    echo Please run setup_env.bat to install dependencies.
    pause
    exit /b 1
)

REM Настройка переменных окружения для больших файлов
echo Setting up environment for large file support...
set STREAMLIT_SERVER_MAX_UPLOAD_SIZE=5000
set STREAMLIT_SERVER_MAX_MESSAGE_SIZE=5000
set STREAMLIT_SERVER_ENABLE_CORS=false
set STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
set STREAMLIT_GLOBAL_DEVELOPMENT_MODE=false

REM Настройка памяти для Python
set PYTHONMALLOC=malloc
set MALLOC_ARENA_MAX=2

REM Создание конфигурационного файла Streamlit
echo Creating Streamlit configuration...
if not exist ".streamlit" mkdir .streamlit

echo [server] > .streamlit\config.toml
echo maxUploadSize = 5000 >> .streamlit\config.toml
echo maxMessageSize = 5000 >> .streamlit\config.toml
echo enableCORS = false >> .streamlit\config.toml
echo enableXsrfProtection = false >> .streamlit\config.toml
echo. >> .streamlit\config.toml
echo [browser] >> .streamlit\config.toml
echo gatherUsageStats = false >> .streamlit\config.toml
echo. >> .streamlit\config.toml
echo [global] >> .streamlit\config.toml
echo developmentMode = false >> .streamlit\config.toml
echo unitTest = false >> .streamlit\config.toml

REM Выбор файла приложения
echo.
echo Select application to run:
echo 1. Full application (app.py)
echo 2. Small application (app_small.py)
echo 3. Version 070625 (app_070625.py)
echo.
set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" (
    set APP_FILE=app.py
    set APP_NAME=Full Application
) else if "%choice%"=="2" (
    set APP_FILE=app_small.py
    set APP_NAME=Small Application
) else if "%choice%"=="3" (
    set APP_FILE=app_070625.py
    set APP_NAME=Version 070625
) else (
    echo Invalid choice. Using default app.py
    set APP_FILE=app.py
    set APP_NAME=Full Application
)

echo.
echo ===================================
echo Starting %APP_NAME%
echo ===================================
echo.
echo Configuration:
echo - Max upload size: 5000 MB (5 GB)
echo - Max message size: 5000 MB
echo - Memory optimization: Enabled
echo - CORS: Disabled (for local use)
echo.
echo The application will open in your default browser.
echo Press Ctrl+C to stop the server.
echo.

REM Поиск свободного порта
echo Searching for available port...
for /L %%i in (8501,1,8510) do (
    netstat -an | findstr ":%%i " >nul
    if errorlevel 1 (
        set "AVAILABLE_PORT=%%i"
        goto :found_port
    )
)

REM Если порты с 8501 по 8510 заняты, используем случайный порт
echo Ports 8501-8510 are busy, using random port...
set "AVAILABLE_PORT=0"

:found_port
echo Using port: %AVAILABLE_PORT%
echo.

REM Запуск приложения
if "%AVAILABLE_PORT%"=="0" (
    streamlit run %APP_FILE% --server.address=localhost
) else (
    streamlit run %APP_FILE% --server.port=%AVAILABLE_PORT% --server.address=localhost
)

echo.
echo Application stopped.
pause
