#!/bin/bash

# Активация окружения
conda activate translation-evaluator

# Запуск тестов с покрытием
pytest --cov=translation_utils tests/ --cov-report=html

# Запуск линтеров
echo "Running black..."
black src/ tests/

echo "Running isort..."
isort src/ tests/

echo "Running flake8..."
flake8 src/ tests/

echo "Running mypy..."
mypy src/ tests/

echo "All tests and checks completed!" 