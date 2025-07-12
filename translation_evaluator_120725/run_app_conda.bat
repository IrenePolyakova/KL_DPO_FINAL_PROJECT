@echo off
chcp 65001 > nul
echo ===================================
echo Запуск приложения (Conda окружение)
echo ===================================

REM Быстрая проверка наличия conda
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo ОШИБКА: Conda не найдена в PATH
    echo Пожалуйста, установите Miniconda или Anaconda
    echo Или используйте Anaconda Prompt вместо обычной командной строки
    pause
    exit /b 1
)

echo [OK] Conda найдена

REM Быстрая проверка существования окружения
echo Проверка окружения translation_env...
conda info --envs 2>nul | findstr /C:"translation_env" >nul
if %errorlevel% neq 0 (
    echo ОШИБКА: Окружение translation_env не найдено
    echo Запустите setup_conda_env.bat для создания окружения
    pause
    exit /b 1
)

echo [OK] Окружение translation_env найдено

REM Активация окружения
echo Активация окружения translation_env...
call conda activate translation_env
if %errorlevel% neq 0 (
    echo ОШИБКА: Не удалось активировать окружение
    echo Попробуйте пересоздать окружение через setup_conda_env.bat
    pause
    exit /b 1
)

echo [OK] Окружение активировано

REM Проверка наличия приложения
if not exist "app.py" (
    echo ОШИБКА: Файл app.py не найден
    pause
    exit /b 1
)

REM Установка переменных окружения для оптимизации
echo Настройка переменных окружения...
set STREAMLIT_SERVER_MAX_UPLOAD_SIZE=5000
set STREAMLIT_SERVER_ENABLE_CORS=false
set STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
set TOKENIZERS_PARALLELISM=false

REM Быстрый выбор порта
echo Проверка портов...
set PORT=8501
powershell -Command "Test-NetConnection -ComputerName localhost -Port 8501 -InformationLevel Quiet -WarningAction SilentlyContinue" 2>nul | findstr "True" >nul
if %errorlevel% neq 0 (
    set PORT=8501
    goto :found_port
)
powershell -Command "Test-NetConnection -ComputerName localhost -Port 8502 -InformationLevel Quiet -WarningAction SilentlyContinue" 2>nul | findstr "True" >nul
if %errorlevel% neq 0 (
    set PORT=8502
    goto :found_port
)
set PORT=8503

:found_port
echo [OK] Используемый порт: %PORT%
echo.

REM Вывод информации о системе
echo ===================================
echo ИНФОРМАЦИЯ О СИСТЕМЕ
echo ===================================
echo [ПРОЦЕСС] Сбор информации о системе...
echo [PYTHON]
python --version 2>nul || echo Не удалось получить версию Python
echo [PYTORCH]
python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>nul || echo Не удалось получить версию PyTorch
echo [CUDA]
python -c "import torch; print(f'CUDA доступна: {\"Да\" if torch.cuda.is_available() else \"Нет (CPU режим)\"}')" 2>nul || echo Не удалось проверить CUDA
echo [GPU]
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU режим\"}')" 2>nul || echo Не удалось получить информацию о GPU
echo.

REM Запуск приложения
echo ===================================
echo ЗАПУСК ПРИЛОЖЕНИЯ
echo ===================================
echo [ПРОЦЕСС] Запуск Translation Evaluator на порту %PORT%...
echo [INFO] Приложение будет доступно по адресу: http://localhost:%PORT%
echo [INFO] Для остановки нажмите Ctrl+C
echo [INFO] Логи приложения будут отображаться ниже:
echo.
echo ===================================
echo ЛОГИ ПРИЛОЖЕНИЯ
echo ===================================

streamlit run app.py --server.port=%PORT% --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false --browser.gatherUsageStats=false

REM Деактивация окружения при завершении
echo.
echo [ПРОЦЕСС] Деактивация окружения...
call conda deactivate
pause
