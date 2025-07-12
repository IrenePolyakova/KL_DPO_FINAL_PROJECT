@echo off
chcp 65001 > nul
echo ===================================
echo   Настройка окружения через Conda
echo ===================================

REM Проверка наличия conda
echo Проверка наличия conda...
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo ОШИБКА: Conda не найдена в PATH
    echo Пожалуйста, установите Miniconda или Anaconda
    echo Скачать можно с: https://docs.conda.io/en/latest/miniconda.html
    echo.
    echo Также проверьте что conda добавлена в PATH:
    echo 1. Откройте Anaconda Prompt
    echo 2. Выполните: conda init cmd.exe
    echo 3. Перезапустите командную строку
    pause
    exit /b 1
)

echo [OK] Conda найдена

REM Удаление старого окружения если существует
echo Удаление старого окружения если существует...
conda env remove -n translation_env -y 2>nul

REM Создание нового окружения с Python 3.10
echo Создание нового окружения с Python 3.10...
conda create -n translation_env python=3.10 -y
if %errorlevel% neq 0 (
    echo ОШИБКА: Не удалось создать окружение
    pause
    exit /b 1
)

REM Активация окружения
echo Активация окружения...
call conda activate translation_env
if %errorlevel% neq 0 (
    echo ОШИБКА: Не удалось активировать окружение
    pause
    exit /b 1
)

REM Проверка наличия NVIDIA GPU
echo Проверка наличия GPU...
where nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    nvidia-smi -L >nul 2>&1
    if %errorlevel% equ 0 (
        echo [OK] NVIDIA GPU обнаружен - устанавливаем PyTorch с CUDA
        echo Установка PyTorch с CUDA...
        conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
        if %errorlevel% neq 0 (
            echo ПРЕДУПРЕЖДЕНИЕ: Не удалось установить PyTorch с CUDA, устанавливаем CPU версию
            conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
        )
    ) else (
        echo [INFO] NVIDIA GPU найден, но драйверы не работают - устанавливаем CPU версию
        conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
    )
) else (
    echo [INFO] NVIDIA GPU не найден - устанавливаем CPU версию
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
)

REM Установка основных зависимостей
echo Установка основных зависимостей...
conda install numpy pandas scipy scikit-learn matplotlib seaborn jupyter -y

REM Обновление pip
echo Обновление pip...
python -m pip install --upgrade pip

REM Установка специфических пакетов для перевода
echo Установка пакетов для машинного перевода...
pip install streamlit>=1.24.0
pip install transformers>=4.30.0
pip install sacrebleu>=2.3.1
pip install python-docx>=0.8.11
pip install safetensors>=0.3.1
pip install sentencepiece>=0.1.99
pip install protobuf>=3.20.0
pip install reportlab>=4.0.4
pip install langdetect>=1.0.9
pip install nest-asyncio==1.5.6
pip install accelerate>=0.20.0
pip install pydantic>=2.0.0
pip install openpyxl>=3.0.0
pip install xlrd>=2.0.0
pip install xlsxwriter>=3.0.0
pip install chardet>=5.0.0
pip install python-dotenv>=1.0.0
pip install tqdm>=4.65.0

echo.
echo ===================================
echo   Проверка установки
echo ===================================

REM Проверка Python
echo Проверка Python:
python --version
python -c "import sys; print(f'Python path: {sys.executable}')"

REM Проверка PyTorch и CUDA
echo.
echo Проверка PyTorch:
python -c "import torch; print(f'PyTorch версия: {torch.__version__}')"
python -c "import torch; print(f'CUDA доступна: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA версия: {torch.version.cuda if torch.cuda.is_available() else \"Не доступна\"}')"
if %errorlevel% equ 0 (
    python -c "import torch; print(f'GPU устройств: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"
    if %errorlevel% equ 0 (
        python -c "import torch; print(f'GPU модель: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"Нет GPU\"}')"
    )
)

REM Проверка основных библиотек
echo.
echo Проверка основных библиотек:
python -c "import transformers; print(f'Transformers: {transformers.__version__}')" 2>nul && echo [OK] Transformers || echo [ERROR] Transformers
python -c "import streamlit; print(f'Streamlit: {streamlit.__version__}')" 2>nul && echo [OK] Streamlit || echo [ERROR] Streamlit
python -c "import accelerate; print('[OK] Accelerate')" 2>nul || echo [ERROR] Accelerate
python -c "import pydantic; print(f'Pydantic: {pydantic.__version__}')" 2>nul && echo [OK] Pydantic || echo [ERROR] Pydantic

echo.
echo ===================================
echo   Окружение готово!
echo ===================================
echo.
echo Для активации окружения используйте:
echo conda activate translation_env
echo.
echo Для деактивации:
echo conda deactivate
echo.
echo Информация об окружении:
conda info --envs
echo.
pause
