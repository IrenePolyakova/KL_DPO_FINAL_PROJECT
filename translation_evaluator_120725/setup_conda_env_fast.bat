@echo off
chcp 65001 > nul
echo ===================================
echo Быстрая установка Conda окружения
echo ===================================

REM Проверка наличия conda
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo ОШИБКА: Conda не найдена в PATH
    echo Пожалуйста, установите Miniconda или Anaconda
    echo Или используйте Anaconda Prompt вместо обычной командной строки
    pause
    exit /b 1
)

echo [OK] Conda найдена

REM Удаление существующего окружения (если есть)
echo Удаление существующего окружения translation_env...
echo [ПРОЦЕСС] Проверка и удаление старого окружения...
conda env remove -n translation_env -y
echo.

REM Создание нового окружения с Python 3.9
echo ===================================
echo СОЗДАНИЕ ОКРУЖЕНИЯ
echo ===================================
echo [ПРОЦЕСС] Создание нового окружения translation_env с Python 3.9...
echo [INFO] Этот процесс может занять несколько минут...
conda create -n translation_env python=3.9 -y
if %errorlevel% neq 0 (
    echo [ОШИБКА] Не удалось создать conda окружение
    pause
    exit /b 1
)

echo [OK] Окружение создано успешно
echo.

REM Активация окружения
echo ===================================
echo АКТИВАЦИЯ ОКРУЖЕНИЯ
echo ===================================
echo [ПРОЦЕСС] Активация окружения translation_env...
call conda activate translation_env
if %errorlevel% neq 0 (
    echo [ОШИБКА] Не удалось активировать окружение
    pause
    exit /b 1
)

echo [OK] Окружение активировано успешно
echo.

REM Установка основных зависимостей через conda (быстрее)
echo ===================================
echo УСТАНОВКА ОСНОВНЫХ ЗАВИСИМОСТЕЙ
echo ===================================
echo [ПРОЦЕСС] Установка основных зависимостей через conda...
echo [INFO] Устанавливаются: streamlit, pandas, numpy
conda install -c conda-forge streamlit pandas numpy -y
if %errorlevel% neq 0 (
    echo [ОШИБКА] Не удалось установить основные зависимости
    pause
    exit /b 1
)

echo [OK] Основные зависимости установлены успешно
echo.

REM Установка PyTorch CPU (для быстрой установки)
echo ===================================
echo УСТАНОВКА PYTORCH
echo ===================================
echo [ПРОЦЕСС] Установка PyTorch CPU версии...
echo [INFO] Устанавливаются: pytorch, torchvision, torchaudio (CPU)
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
if %errorlevel% neq 0 (
    echo [ОШИБКА] Не удалось установить PyTorch
    pause
    exit /b 1
)

echo [OK] PyTorch установлен успешно
echo.

REM Установка дополнительных зависимостей через pip
echo ===================================
echo УСТАНОВКА ДОПОЛНИТЕЛЬНЫХ ЗАВИСИМОСТЕЙ
echo ===================================
echo [ПРОЦЕСС] Установка дополнительных зависимостей через pip...
echo [INFO] Устанавливаются: transformers, datasets, sacrebleu, nltk, scikit-learn, matplotlib, seaborn, plotly
pip install transformers datasets sacrebleu nltk scikit-learn matplotlib seaborn plotly
pip install transformers datasets sacrebleu nltk scikit-learn matplotlib seaborn plotly
if %errorlevel% neq 0 (
    echo [ОШИБКА] Не удалось установить дополнительные зависимости
    pause
    exit /b 1
)

echo [OK] Дополнительные зависимости установлены успешно
echo.

REM Проверка установки
echo ===================================
echo ПРОВЕРКА УСТАНОВКИ
echo ===================================
echo [ПРОЦЕСС] Проверка успешности установки пакетов...
echo.
echo [STREAMLIT]
python -c "import streamlit; print('Streamlit:', streamlit.__version__)"
echo [PYTORCH]
python -c "import torch; print('PyTorch:', torch.__version__)"
echo [TRANSFORMERS]
python -c "import transformers; print('Transformers:', transformers.__version__)"
echo [DATASETS]
python -c "import datasets; print('Datasets:', datasets.__version__)" 2>nul || echo "Datasets: не установлен"
echo [SACREBLEU]
python -c "import sacrebleu; print('SacreBLEU:', sacrebleu.__version__)" 2>nul || echo "SacreBLEU: не установлен"
echo.

echo ===================================
echo УСТАНОВКА ЗАВЕРШЕНА УСПЕШНО!
echo ===================================
echo Окружение translation_env готово к использованию
echo.
echo [CUDA СТАТУС]
python -c "import torch; print('CUDA доступна:', 'Да' if torch.cuda.is_available() else 'Нет (используется CPU)')"
echo.
echo [ИНСТРУКЦИИ]
echo Для запуска приложения используйте:
echo - run_app_conda.bat (с проверками)
echo - run_app_conda_fast.bat (быстрый запуск)
echo.

pause
echo ===================================
echo.
echo Для активации окружения используйте:
echo conda activate translation_env
echo.
echo Для запуска приложения используйте:
echo run_app_conda_fast.bat
echo.
call conda deactivate
pause
