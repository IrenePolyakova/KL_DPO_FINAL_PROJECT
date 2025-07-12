@echo off
echo ===================================
echo  Ultra Performance Translation App
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

REM Расширенные настройки производительности
echo Setting up ultra-performance environment...

REM Streamlit настройки для максимальной производительности
set STREAMLIT_SERVER_MAX_UPLOAD_SIZE=10000
set STREAMLIT_SERVER_MAX_MESSAGE_SIZE=10000
set STREAMLIT_SERVER_ENABLE_CORS=false
set STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
set STREAMLIT_GLOBAL_DEVELOPMENT_MODE=false
set STREAMLIT_SERVER_RUN_ON_SAVE=false

REM PyTorch максимальная производительность
set TORCH_CUDNN_V8_API_ENABLED=1
set CUDA_LAUNCH_BLOCKING=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,roundup_power2_divisions:8
set PYTORCH_NO_CUDA_MEMORY_CACHING=0
set CUDA_CACHE_DISABLE=0

REM CPU оптимизация (максимальное использование ядер)
for /f "tokens=2 delims==" %%i in ('wmic cpu get NumberOfLogicalProcessors /value ^| findstr "="') do set CPU_CORES=%%i
set /a OPTIMAL_THREADS=%CPU_CORES%-1
if %OPTIMAL_THREADS% LSS 1 set OPTIMAL_THREADS=1

set OMP_NUM_THREADS=%OPTIMAL_THREADS%
set MKL_NUM_THREADS=%OPTIMAL_THREADS%
set NUMEXPR_NUM_THREADS=%OPTIMAL_THREADS%
set OPENBLAS_NUM_THREADS=%OPTIMAL_THREADS%

REM Память и кэширование
set PYTHONMALLOC=malloc
set MALLOC_ARENA_MAX=2
set PYTHONHASHSEED=0

REM Оптимизированные кэши
set TORCH_HOME=.\torch_cache_ultra
set TRANSFORMERS_CACHE=.\transformers_cache_ultra
set HF_HOME=.\huggingface_cache_ultra
set HF_DATASETS_CACHE=.\datasets_cache_ultra

REM Создание оптимизированных кэш-директорий
if not exist "torch_cache_ultra" mkdir torch_cache_ultra
if not exist "transformers_cache_ultra" mkdir transformers_cache_ultra
if not exist "huggingface_cache_ultra" mkdir huggingface_cache_ultra
if not exist "datasets_cache_ultra" mkdir datasets_cache_ultra

REM Проверка GPU
echo Checking GPU availability...
python -c "
import torch
cuda_available = torch.cuda.is_available()
print(f'GPU Support: {\"YES\" if cuda_available else \"NO (CPU only)\"}')
if cuda_available:
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

REM Создание расширенного конфигурационного файла
if not exist ".streamlit" mkdir .streamlit

echo [server] > .streamlit\config.toml
echo maxUploadSize = 10000 >> .streamlit\config.toml
echo maxMessageSize = 10000 >> .streamlit\config.toml
echo enableCORS = false >> .streamlit\config.toml
echo enableXsrfProtection = false >> .streamlit\config.toml
echo headless = false >> .streamlit\config.toml
echo runOnSave = false >> .streamlit\config.toml
echo fileWatcherType = "none" >> .streamlit\config.toml
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
echo primaryColor = "#FF4B4B" >> .streamlit\config.toml
echo backgroundColor = "#FFFFFF" >> .streamlit\config.toml
echo secondaryBackgroundColor = "#F0F2F6" >> .streamlit\config.toml
echo textColor = "#262730" >> .streamlit\config.toml
echo. >> .streamlit\config.toml
echo [runner] >> .streamlit\config.toml
echo magicEnabled = false >> .streamlit\config.toml
echo installTracer = false >> .streamlit\config.toml
echo fixMatplotlib = false >> .streamlit\config.toml

REM Поиск свободного порта
echo Searching for optimal port...
set "AVAILABLE_PORT="

for %%p in (8501 8502 8503 8504 8505) do (
    netstat -an | findstr ":%%p " >nul 2>&1
    if errorlevel 1 (
        set "AVAILABLE_PORT=%%p"
        goto :found_port
    )
)

for /L %%i in (8506,1,8520) do (
    netstat -an | findstr ":%%i " >nul 2>&1
    if errorlevel 1 (
        set "AVAILABLE_PORT=%%i"
        goto :found_port
    )
)

set "AVAILABLE_PORT=0"

:found_port
echo.
echo ===================================
echo Ultra Performance Mode Activated
echo ===================================
echo.
echo Performance Settings:
echo - Max upload size: 10000 MB (10 GB)
echo - CPU threads: %OPTIMAL_THREADS% (of %CPU_CORES% cores)
echo - Memory optimization: Advanced
echo - GPU acceleration: Auto-detected
echo - File watching: Disabled
echo - Magic commands: Disabled
echo - Port: %AVAILABLE_PORT%
echo.

echo Starting ultra-performance application...
echo.

REM Запуск с максимальной производительностью
if "%AVAILABLE_PORT%"=="0" (
    python -O -u -m streamlit run app.py ^
        --server.address=localhost ^
        --server.headless=false ^
        --server.runOnSave=false ^
        --server.fileWatcherType=none ^
        --browser.gatherUsageStats=false ^
        --global.developmentMode=false ^
        --runner.magicEnabled=false ^
        --runner.installTracer=false ^
        --runner.fixMatplotlib=false
) else (
    python -O -u -m streamlit run app.py ^
        --server.port=%AVAILABLE_PORT% ^
        --server.address=localhost ^
        --server.headless=false ^
        --server.runOnSave=false ^
        --server.fileWatcherType=none ^
        --browser.gatherUsageStats=false ^
        --global.developmentMode=false ^
        --runner.magicEnabled=false ^
        --runner.installTracer=false ^
        --runner.fixMatplotlib=false
)

echo.
echo Ultra-performance application stopped.
pause
