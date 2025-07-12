@echo off
echo ===================================
echo  High Performance Translation App
echo ===================================

REM Проверка активации виртуального окружения
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup_env.bat first.
    pause
    exit /b 1
)

REM Активация виртуального окружения
call venv\Scripts\activate.bat

REM Расширенные настройки для больших файлов и производительности
echo Setting up high-performance environment...

REM Streamlit настройки
set STREAMLIT_SERVER_MAX_UPLOAD_SIZE=8000
set STREAMLIT_SERVER_MAX_MESSAGE_SIZE=8000
set STREAMLIT_SERVER_ENABLE_CORS=false
set STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
set STREAMLIT_GLOBAL_DEVELOPMENT_MODE=false

REM Python настройки памяти
set PYTHONMALLOC=malloc
set MALLOC_ARENA_MAX=2
set PYTHONHASHSEED=0
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4

REM PyTorch настройки
set TORCH_HOME=.\torch_cache
set TRANSFORMERS_CACHE=.\transformers_cache
set HF_HOME=.\huggingface_cache

REM Создание кэш-директорий
if not exist "torch_cache" mkdir torch_cache
if not exist "transformers_cache" mkdir transformers_cache
if not exist "huggingface_cache" mkdir huggingface_cache

REM Создание расширенного конфигурационного файла
if not exist ".streamlit" mkdir .streamlit

echo [server] > .streamlit\config.toml
echo maxUploadSize = 8000 >> .streamlit\config.toml
echo maxMessageSize = 8000 >> .streamlit\config.toml
echo enableCORS = false >> .streamlit\config.toml
echo enableXsrfProtection = false >> .streamlit\config.toml
echo headless = false >> .streamlit\config.toml
echo runOnSave = false >> .streamlit\config.toml
echo. >> .streamlit\config.toml
echo [browser] >> .streamlit\config.toml
echo gatherUsageStats = false >> .streamlit\config.toml
echo serverAddress = "localhost" >> .streamlit\config.toml
echo. >> .streamlit\config.toml
echo [global] >> .streamlit\config.toml
echo developmentMode = false >> .streamlit\config.toml
echo unitTest = false >> .streamlit\config.toml
echo. >> .streamlit\config.toml
echo [theme] >> .streamlit\config.toml
echo primaryColor = "#FF6B6B" >> .streamlit\config.toml
echo backgroundColor = "#FFFFFF" >> .streamlit\config.toml
echo secondaryBackgroundColor = "#F0F2F6" >> .streamlit\config.toml
echo textColor = "#262730" >> .streamlit\config.toml

echo.
echo ===================================
echo High Performance Mode Activated
echo ===================================
echo.
echo Performance Settings:
echo - Max upload size: 8000 MB (8 GB)
echo - Max message size: 8000 MB
echo - Memory optimization: Advanced
echo - PyTorch cache: Local
echo - Transformers cache: Local
echo - Threading: Optimized
echo.
echo Starting application...
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
echo Application will be available at: http://localhost:%AVAILABLE_PORT%
echo.

REM Запуск с оптимизированными параметрами
if "%AVAILABLE_PORT%"=="0" (
    python -u -m streamlit run app.py ^
        --server.address=localhost ^
        --server.headless=false ^
        --server.runOnSave=false ^
        --browser.gatherUsageStats=false ^
        --global.developmentMode=false
) else (
    python -u -m streamlit run app.py ^
        --server.port=%AVAILABLE_PORT% ^
        --server.address=localhost ^
        --server.headless=false ^
        --server.runOnSave=false ^
        --browser.gatherUsageStats=false ^
        --global.developmentMode=false
)

echo.
echo Application stopped.
pause
