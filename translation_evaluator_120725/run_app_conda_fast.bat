@echo off
chcp 65001 > nul
echo ===================================
echo Быстрый запуск приложения (Conda)
echo ===================================

REM Активация окружения translation_env
echo [ПРОЦЕСС] Активация окружения translation_env...
call conda activate translation_env 2>nul
if %errorlevel% neq 0 (
    echo [ОШИБКА] Не удалось активировать окружение translation_env
    echo [РЕШЕНИЕ] Запустите setup_conda_env_fast.bat для создания окружения
    pause
    exit /b 1
)

echo [OK] Окружение активировано успешно

REM Установка переменных окружения
echo [ПРОЦЕСС] Настройка переменных окружения для оптимизации...
set STREAMLIT_SERVER_MAX_UPLOAD_SIZE=5000
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
set TOKENIZERS_PARALLELISM=false
echo [OK] Переменные окружения настроены

REM Быстрый запуск на порту 8501
echo.
echo ===================================
echo ЗАПУСК ПРИЛОЖЕНИЯ
echo ===================================
echo [ПРОЦЕСС] Запуск Translation Evaluator на порту 8501...
echo [INFO] Приложение будет доступно по адресу: http://localhost:8501
echo [INFO] Для остановки нажмите Ctrl+C
echo [INFO] Логи приложения будут отображаться ниже:
echo.
echo ===================================
echo ЛОГИ ПРИЛОЖЕНИЯ
echo ===================================

streamlit run app.py --server.port=8501 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false --browser.gatherUsageStats=false
