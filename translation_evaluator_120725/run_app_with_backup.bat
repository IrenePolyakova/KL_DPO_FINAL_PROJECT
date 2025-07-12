@echo off
echo ===================================
echo  Translation App with Backup System
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

REM Установка системы резервного копирования
echo Setting up backup system...
call setup_translation_backup.bat > nul 2>&1

REM Настройка переменных окружения для резервного копирования
echo Setting up backup environment...
set TRANSLATION_BACKUP_ENABLED=1
set TRANSLATION_AUTO_SAVE=1
set TRANSLATION_RESUME_ON_ERROR=1
set STREAMLIT_BACKUP_INTEGRATION=1

REM Стандартные настройки производительности
set STREAMLIT_SERVER_MAX_UPLOAD_SIZE=8000
set STREAMLIT_SERVER_MAX_MESSAGE_SIZE=8000
set STREAMLIT_SERVER_ENABLE_CORS=false
set STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
set STREAMLIT_GLOBAL_DEVELOPMENT_MODE=false

REM PyTorch оптимизации
set TORCH_CUDNN_V8_API_ENABLED=1
set CUDA_LAUNCH_BLOCKING=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

REM CPU оптимизация
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4

REM Создание конфигурационного файла Streamlit с поддержкой резервных копий
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
echo. >> .streamlit\config.toml
echo [global] >> .streamlit\config.toml
echo developmentMode = false >> .streamlit\config.toml
echo. >> .streamlit\config.toml
echo [theme] >> .streamlit\config.toml
echo primaryColor = "#00C851" >> .streamlit\config.toml
echo backgroundColor = "#FFFFFF" >> .streamlit\config.toml
echo secondaryBackgroundColor = "#F0F2F6" >> .streamlit\config.toml
echo textColor = "#262730" >> .streamlit\config.toml

REM Поиск свободного порта
echo Searching for available port...
set "AVAILABLE_PORT="

for %%p in (8501 8502 8503 8504 8505) do (
    netstat -an | findstr ":%%p " >nul 2>&1
    if errorlevel 1 (
        set "AVAILABLE_PORT=%%p"
        goto :found_port
    )
)

for /L %%i in (8506,1,8515) do (
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
echo Translation App with Backup System
echo ===================================
echo.
echo Features:
echo - ✅ Automatic backup of translation results
echo - ✅ Resume interrupted translations
echo - ✅ Model-specific result preservation
echo - ✅ Session recovery system
echo - ✅ Progress tracking and auto-save
echo - ✅ Error recovery with partial results
echo.
echo Configuration:
echo - Max upload size: 8000 MB (8 GB)
echo - Backup system: ENABLED
echo - Auto-save: Every 10 translations
echo - Resume on error: ENABLED
if "%AVAILABLE_PORT%"=="0" (
    echo - Port: Automatic selection
) else (
    echo - Port: %AVAILABLE_PORT%
    echo - URL: http://localhost:%AVAILABLE_PORT%
)
echo.
echo Backup Features:
echo - 💾 Saves results from each model separately
echo - 🔄 Resumes from last checkpoint on interruption
echo - 📊 Tracks translation progress in real-time
echo - 🗂️ Organizes backups by session
echo - 🧹 Automatic cleanup of old backups
echo.

echo Starting application with backup system...
echo.

REM Запуск приложения с поддержкой резервного копирования
if "%AVAILABLE_PORT%"=="0" (
    streamlit run app.py --server.address=localhost
) else (
    streamlit run app.py --server.port=%AVAILABLE_PORT% --server.address=localhost
)

echo.
echo Translation application with backup system stopped.
echo.
echo Backup files are preserved in the 'translation_backups' directory.
echo You can resume interrupted translations by restarting the application.
echo.
pause
