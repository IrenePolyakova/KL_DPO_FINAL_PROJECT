@echo off
chcp 65001 > nul
echo ===================================
echo Универсальная установка окружения Translation Evaluator
echo ===================================

REM Проверка наличия conda
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo [ПРЕДУПРЕЖДЕНИЕ] Conda не найдена. Будет создано venv окружение.
    goto SETUP_VENV
) else (
    echo [OK] Conda найдена
    goto SETUP_CONDA
)

:SETUP_CONDA
echo.
echo ===================================
echo СОЗДАНИЕ CONDA ОКРУЖЕНИЯ
echo ===================================
echo [ПРОЦЕСС] Создание окружения из environment.yml...
echo [INFO] Используется Python 3.9 + оптимизированные зависимости

REM Удаление существующего окружения
conda env remove -n translation_env -y >nul 2>&1

REM Создание окружения из YAML файла
conda env create -f environment.yml
if %errorlevel% neq 0 (
    echo [ОШИБКА] Не удалось создать conda окружение
    echo [INFO] Пробуем альтернативный способ...
    goto SETUP_CONDA_MANUAL
)

echo [OK] Conda окружение создано успешно
goto VERIFY_INSTALLATION

:SETUP_CONDA_MANUAL
echo.
echo ===================================
echo АЛЬТЕРНАТИВНАЯ УСТАНОВКА CONDA
echo ===================================
echo [ПРОЦЕСС] Создание окружения вручную...

REM Создание базового окружения
conda create -n translation_env python=3.9 -y
if %errorlevel% neq 0 (
    echo [ОШИБКА] Не удалось создать базовое окружение
    goto SETUP_VENV
)

REM Активация окружения
call conda activate translation_env
if %errorlevel% neq 0 (
    echo [ОШИБКА] Не удалось активировать окружение
    goto SETUP_VENV
)

REM Установка основных зависимостей через conda
echo [ПРОЦЕСС] Установка основных пакетов через conda...
conda install -c conda-forge streamlit pandas numpy matplotlib seaborn scikit-learn openpyxl -y
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

REM Установка дополнительных пакетов через pip
echo [ПРОЦЕСС] Установка дополнительных пакетов через pip...
pip install -r requirements.txt

goto VERIFY_INSTALLATION

:SETUP_VENV
echo.
echo ===================================
echo СОЗДАНИЕ VENV ОКРУЖЕНИЯ
echo ===================================
echo [ПРОЦЕСС] Создание виртуального окружения Python...

REM Проверка наличия Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ОШИБКА] Python не найден в PATH
    echo [ИНСТРУКЦИЯ] Установите Python 3.9+ с https://python.org
    pause
    exit /b 1
)

REM Удаление существующего venv
if exist "venv" (
    rmdir /s /q "venv"
)

REM Создание venv
python -m venv venv
if %errorlevel% neq 0 (
    echo [ОШИБКА] Не удалось создать виртуальное окружение
    pause
    exit /b 1
)

REM Активация venv
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ОШИБКА] Не удалось активировать виртуальное окружение
    pause
    exit /b 1
)

REM Обновление pip
echo [ПРОЦЕСС] Обновление pip...
python -m pip install --upgrade pip

REM Установка пакетов
echo [ПРОЦЕСС] Установка пакетов из requirements.txt...
pip install -r requirements.txt

echo [OK] Venv окружение создано и настроено

:VERIFY_INSTALLATION
echo.
echo ===================================
echo ПРОВЕРКА УСТАНОВКИ
echo ===================================
echo [ПРОЦЕСС] Проверка установленных пакетов...

REM Проверка основных пакетов
python -c "import streamlit; print('[OK] Streamlit:', streamlit.__version__)" 2>nul || echo "[ОШИБКА] Streamlit не установлен"
python -c "import torch; print('[OK] PyTorch:', torch.__version__)" 2>nul || echo "[ОШИБКА] PyTorch не установлен"
python -c "import transformers; print('[OK] Transformers:', transformers.__version__)" 2>nul || echo "[ОШИБКА] Transformers не установлен"
python -c "import pandas; print('[OK] Pandas:', pandas.__version__)" 2>nul || echo "[ОШИБКА] Pandas не установлен"
python -c "import numpy; print('[OK] NumPy:', numpy.__version__)" 2>nul || echo "[ОШИБКА] NumPy не установлен"

REM Проверка CUDA
echo [CUDA STATUS]
python -c "import torch; print('[CUDA] Доступна:', 'Да' if torch.cuda.is_available() else 'Нет (CPU режим)')" 2>nul || echo "[CUDA] Проверка недоступна"

echo.
echo ===================================
echo УСТАНОВКА ЗАВЕРШЕНА!
echo ===================================

REM Определение типа окружения
where conda >nul 2>&1
if %errorlevel% equ 0 (
    conda env list | findstr /C:"translation_env" >nul
    if %errorlevel% equ 0 (
        echo [ОКРУЖЕНИЕ] Conda окружение translation_env
        echo [АКТИВАЦИЯ] conda activate translation_env
        echo [ЗАПУСК] run_app_conda.bat или run_app_conda_fast.bat
    ) else (
        echo [ОКРУЖЕНИЕ] Venv окружение
        echo [АКТИВАЦИЯ] venv\Scripts\activate.bat
        echo [ЗАПУСК] run_app.bat
    )
) else (
    echo [ОКРУЖЕНИЕ] Venv окружение
    echo [АКТИВАЦИЯ] venv\Scripts\activate.bat
    echo [ЗАПУСК] run_app.bat
)

echo.
echo [КОМАНДЫ ДЛЯ ЗАПУСКА]
echo - Главное меню: start.bat
echo - Быстрый запуск: run_app_conda_fast.bat (если conda) или run_app.bat (если venv)
echo - Подготовка файлов: run_file_preparation.bat
echo - Очистка системы: cleanup_system.bat
echo.

pause
