@echo off
echo ===================================
echo  Translation Performance Optimizer
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

echo Optimizing translation performance...
echo.

REM Проверка доступности CUDA/GPU
echo Checking GPU availability...
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'Current GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('GPU not available - using CPU (slower)')
"

echo.
echo Installing performance optimization packages...

REM Установка оптимизированных библиотек
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --upgrade 2>nul
if %errorlevel% neq 0 (
    echo CUDA version not supported, installing CPU version...
    pip install torch torchvision torchaudio --upgrade
)

REM Установка дополнительных библиотек для ускорения
pip install transformers[torch] --upgrade
pip install accelerate --upgrade
pip install optimum --upgrade

REM Настройка переменных окружения для производительности
echo Setting performance environment variables...

REM PyTorch оптимизации
set TORCH_CUDNN_V8_API_ENABLED=1
set CUDA_LAUNCH_BLOCKING=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

REM Оптимизация памяти
set PYTORCH_NO_CUDA_MEMORY_CACHING=0
set CUDA_CACHE_DISABLE=0

REM Оптимизация CPU
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
set NUMEXPR_NUM_THREADS=4

REM Оптимизация Transformers
set TRANSFORMERS_CACHE=.\transformers_cache_optimized
set HF_HOME=.\huggingface_cache_optimized

REM Создание оптимизированных кэш-директорий
if not exist "transformers_cache_optimized" mkdir transformers_cache_optimized
if not exist "huggingface_cache_optimized" mkdir huggingface_cache_optimized

echo.
echo Performance optimization completed!
echo.
echo Recommendations for faster translation:
echo - Use GPU if available (CUDA supported)
echo - Reduce batch size for large texts
echo - Use smaller models for faster processing
echo - Close other applications to free memory
echo.
pause
