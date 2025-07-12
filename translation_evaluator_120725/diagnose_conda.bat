@echo off
chcp 65001 > nul
echo ===================================
echo  Диагностика системы (Conda)
echo ===================================

REM Проверка conda
echo [1/10] Проверка conda...
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Conda не найдена в PATH
    echo Решение: Установите Miniconda или Anaconda
    echo Или используйте Anaconda Prompt
    goto :end_check
) else (
    echo [OK] Conda найдена
)

REM Проверка окружения
echo [2/10] Проверка окружения translation_env...
conda env list | findstr /C:"translation_env" >nul
if %errorlevel% neq 0 (
    echo [ERROR] Окружение translation_env не найдено
    echo Решение: Запустите setup_conda_env.bat
    goto :end_check
) else (
    echo [OK] Окружение translation_env найдено
)

REM Активация окружения
echo [3/10] Активация окружения...
call conda activate translation_env 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Не удалось активировать окружение
    echo Решение: Пересоздайте окружение
    goto :end_check
) else (
    echo [OK] Окружение активировано
)

REM Проверка Python
echo [4/10] Проверка Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python не работает в окружении
    echo Решение: Пересоздайте окружение
    goto :end_check
) else (
    echo [OK] Python работает
    python --version
)

REM Проверка PyTorch
echo [5/10] Проверка PyTorch...
python -c "import torch; print(f'PyTorch {torch.__version__} работает')" 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] PyTorch не установлен или не работает
    echo Решение: pip install torch torchvision torchaudio
    goto :end_check
) else (
    echo [OK] PyTorch работает
)

REM Проверка CUDA
echo [6/10] Проверка CUDA...
python -c "import torch; print(f'CUDA доступна: {torch.cuda.is_available()}')" 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] Не удалось проверить CUDA
) else (
    python -c "import torch; print(f'CUDA устройств: {torch.cuda.device_count()}')" 2>nul
    python -c "import torch; print(f'GPU модель: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU режим\"}')" 2>nul
)

REM Проверка Transformers
echo [7/10] Проверка Transformers...
python -c "import transformers; print(f'Transformers {transformers.__version__} работает')" 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Transformers не установлен
    echo Решение: pip install transformers
    goto :end_check
) else (
    echo [OK] Transformers работает
)

REM Проверка Streamlit
echo [8/10] Проверка Streamlit...
python -c "import streamlit; print(f'Streamlit {streamlit.__version__} работает')" 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Streamlit не установлен
    echo Решение: pip install streamlit
    goto :end_check
) else (
    echo [OK] Streamlit работает
)

REM Проверка файлов приложения
echo [9/10] Проверка файлов приложения...
if not exist "app.py" (
    echo [ERROR] Файл app.py не найден
    echo Решение: Убедитесь что вы в правильной папке
    goto :end_check
) else (
    echo [OK] Файл app.py найден
)

REM Проверка портов
echo [10/10] Проверка доступных портов...
netstat -an | findstr :8501 >nul
if %errorlevel% equ 0 (
    echo [WARNING] Порт 8501 занят
    echo Решение: Остановите другие приложения или используйте другой порт
) else (
    echo [OK] Порт 8501 свободен
)

echo.
echo ===================================
echo  Дополнительная диагностика
echo ===================================

REM Проверка GPU через nvidia-smi
echo Проверка GPU через nvidia-smi:
nvidia-smi --query-gpu=name,memory.total,memory.used,temperature.gpu --format=csv,noheader 2>nul || echo GPU информация недоступна

REM Проверка памяти
echo.
echo Проверка системной памяти:
wmic OS get TotalVisibleMemorySize,FreePhysicalMemory /value | findstr "=" 2>nul || echo Информация недоступна

REM Проверка места на диске
echo.
echo Проверка места на диске:
for /f "tokens=3" %%i in ('dir /-c ^| findstr /i "bytes free"') do echo Свободно: %%i байт

REM Проверка процессов
echo.
echo Активные процессы Python:
tasklist | findstr /i python 2>nul || echo Процессы Python не найдены

echo.
echo ===================================
echo  Диагностика завершена
echo ===================================

:end_check
call conda deactivate 2>nul
pause
