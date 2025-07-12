@echo off
echo ===================================
echo  Fixing Dependencies Issues
echo ===================================

REM Проверка виртуального окружения
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup_env.bat first.
    pause
    exit /b 1
)

REM Активация виртуального окружения
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing missing accelerate dependency...
pip install accelerate>=0.20.0

echo Installing additional ML dependencies...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo Updating transformers...
pip install transformers>=4.30.0 --upgrade

echo Installing other potentially missing dependencies...
pip install safetensors>=0.3.1
pip install sentencepiece>=0.1.99
pip install protobuf>=3.20.0
pip install pydantic>=2.0.0

echo.
echo ===================================
echo Verifying installations...
echo ===================================

python -c "import torch; print(f'✓ PyTorch {torch.__version__} installed')" 2>nul || echo "✗ PyTorch installation failed"
python -c "import transformers; print(f'✓ Transformers {transformers.__version__} installed')" 2>nul || echo "✗ Transformers installation failed"
python -c "import accelerate; print(f'✓ Accelerate installed')" 2>nul || echo "✗ Accelerate installation failed"
python -c "import streamlit; print(f'✓ Streamlit installed')" 2>nul || echo "✗ Streamlit installation failed"
python -c "import safetensors; print(f'✓ Safetensors installed')" 2>nul || echo "✗ Safetensors installation failed"
python -c "import sentencepiece; print(f'✓ SentencePiece installed')" 2>nul || echo "✗ SentencePiece installation failed"
python -c "import pydantic; print(f'✓ Pydantic installed')" 2>nul || echo "✗ Pydantic installation failed"

echo.
echo ===================================
echo Dependencies fix completed!
echo ===================================
echo.
echo If you still encounter issues, try:
echo 1. Restart your terminal
echo 2. Run the application again
echo 3. Check if your models are compatible
echo.
pause
