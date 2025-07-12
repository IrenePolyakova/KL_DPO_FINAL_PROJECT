@echo off
echo ===================================
echo  Smart Port Selection Launcher
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

REM Проверка установки зависимостей
echo Checking dependencies...
python -c "import streamlit" 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Streamlit not installed!
    echo Please run start.bat and choose option 1 to install dependencies.
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
echo headless = false >> .streamlit\config.toml
echo. >> .streamlit\config.toml
echo [browser] >> .streamlit\config.toml
echo gatherUsageStats = false >> .streamlit\config.toml
echo. >> .streamlit\config.toml
echo [global] >> .streamlit\config.toml
echo developmentMode = false >> .streamlit\config.toml

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

REM Функция поиска свободного порта (улучшенная)
echo Searching for available port...
set "AVAILABLE_PORT="

REM Проверяем предпочтительные порты
for %%p in (8501 8502 8503 8504 8505) do (
    netstat -an | findstr ":%%p " >nul 2>&1
    if errorlevel 1 (
        set "AVAILABLE_PORT=%%p"
        goto :found_port
    )
)

REM Если предпочтительные порты заняты, ищем в расширенном диапазоне
for /L %%i in (8506,1,8520) do (
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
    echo Application will be available at: http://localhost:%AVAILABLE_PORT%
)

echo.
echo Configuration:
echo - Max upload size: 5000 MB (5 GB)
echo - Max message size: 5000 MB
echo - Memory optimization: Enabled
echo - CORS: Disabled (for local use)
echo - Port: %AVAILABLE_PORT%
echo.

REM Запуск приложения с выбранным портом
if "%AVAILABLE_PORT%"=="0" (
    streamlit run %APP_FILE% --server.address=localhost
) else (
    streamlit run %APP_FILE% --server.port=%AVAILABLE_PORT% --server.address=localhost
)

echo.
echo Application stopped.
pause
