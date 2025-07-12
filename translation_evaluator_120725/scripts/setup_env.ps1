# Создание и активация conda окружения
conda create -n translation-evaluator python=3.8 -y
conda activate translation-evaluator

# Установка зависимостей
pip install -r requirements.txt

# Установка дополнительных зависимостей для разработки
pip install -e ".[dev]"

# Проверка установки
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

# Запуск тестов
pytest

Write-Host "Setup completed!" 