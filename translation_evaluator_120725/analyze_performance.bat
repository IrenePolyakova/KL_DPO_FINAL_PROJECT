@echo off
echo ===================================
echo  Translation Performance Analyzer
echo ===================================

REM Проверка виртуального окружения
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo Analyzing translation performance...
echo.

python -c "
import torch
import sys
import psutil
import platform
import time
from pathlib import Path

print('=' * 60)
print('SYSTEM PERFORMANCE ANALYSIS')
print('=' * 60)

# Системная информация
print(f'Platform: {platform.platform()}')
print(f'Python version: {sys.version.split()[0]}')
print(f'CPU cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical')
print(f'RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB total, {psutil.virtual_memory().available / (1024**3):.1f} GB available')

# GPU информация
print()
print('GPU ANALYSIS:')
print('-' * 30)
if torch.cuda.is_available():
    print(f'✓ CUDA available: {torch.version.cuda}')
    print(f'✓ GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        gpu_props = torch.cuda.get_device_properties(i)
        gpu_memory = gpu_props.total_memory / (1024**3)
        print(f'  GPU {i}: {gpu_props.name} ({gpu_memory:.1f} GB)')
        
        # Проверка доступной памяти GPU
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        print(f'    Memory: {allocated:.1f} GB allocated, {reserved:.1f} GB reserved')
else:
    print('✗ CUDA not available - translations will use CPU (slower)')
    print('  Consider installing CUDA for GPU acceleration')

# PyTorch информация
print()
print('PYTORCH ANALYSIS:')
print('-' * 30)
print(f'PyTorch version: {torch.__version__}')

try:
    import transformers
    print(f'Transformers version: {transformers.__version__}')
except ImportError:
    print('✗ Transformers not installed')

try:
    import accelerate
    print(f'Accelerate version: {accelerate.__version__}')
except ImportError:
    print('✗ Accelerate not installed (recommended for performance)')

# Тест производительности
print()
print('PERFORMANCE BENCHMARK:')
print('-' * 30)

# CPU тест
start_time = time.time()
x = torch.randn(1000, 1000)
y = torch.mm(x, x.t())
cpu_time = time.time() - start_time
print(f'CPU matrix multiplication (1000x1000): {cpu_time:.3f}s')

# GPU тест (если доступен)
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    x_gpu = torch.randn(1000, 1000, device=device)
    torch.cuda.synchronize()
    start_time = time.time()
    y_gpu = torch.mm(x_gpu, x_gpu.t())
    torch.cuda.synchronize()
    gpu_time = time.time() - start_time
    speedup = cpu_time / gpu_time
    print(f'GPU matrix multiplication (1000x1000): {gpu_time:.3f}s')
    print(f'GPU speedup: {speedup:.1f}x faster than CPU')

# Рекомендации по оптимизации
print()
print('OPTIMIZATION RECOMMENDATIONS:')
print('-' * 40)

if not torch.cuda.is_available():
    print('🚀 CRITICAL: Install CUDA for GPU acceleration')
    print('   - Download from: https://developer.nvidia.com/cuda-toolkit')
    print('   - Or run: optimize_performance.bat')

if psutil.virtual_memory().available < 8 * (1024**3):
    print('⚠️  WARNING: Low available RAM (< 8GB)')
    print('   - Close other applications')
    print('   - Consider upgrading RAM')

cpu_percent = psutil.cpu_percent(interval=1)
if cpu_percent > 80:
    print('⚠️  WARNING: High CPU usage detected')
    print('   - Close unnecessary programs')

# Кэш анализ
cache_dirs = [
    'torch_cache', 'torch_cache_ultra',
    'transformers_cache', 'transformers_cache_ultra', 'transformers_cache_optimized',
    'huggingface_cache', 'huggingface_cache_ultra', 'huggingface_cache_optimized'
]

total_cache_size = 0
for cache_dir in cache_dirs:
    if Path(cache_dir).exists():
        cache_size = sum(f.stat().st_size for f in Path(cache_dir).rglob('*') if f.is_file())
        total_cache_size += cache_size
        
if total_cache_size > 0:
    print(f'📁 Cache size: {total_cache_size / (1024**3):.1f} GB')
    if total_cache_size > 5 * (1024**3):
        print('   - Consider running cleanup_system.bat to free space')

print()
print('PERFORMANCE TIPS:')
print('-' * 20)
print('✓ Use smaller models for faster translation')
print('✓ Reduce batch size if running out of memory')
print('✓ Use GPU acceleration when available')
print('✓ Close other applications during translation')
print('✓ Use SSD storage for better I/O performance')
print('✓ Run optimize_performance.bat for auto-optimization')

print()
print('=' * 60)
"

echo.
echo Analysis completed!
echo.
echo To optimize performance automatically, run:
echo   optimize_performance.bat
echo.
echo To launch ultra-performance mode:
echo   run_app_ultra.bat
echo.
pause
