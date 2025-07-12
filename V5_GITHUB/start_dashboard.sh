#!/bin/bash

echo "🚀 Запуск MarianMT Training Dashboard"
echo "======================================"

# Проверка установки Streamlit
if ! command -v streamlit &> /dev/null; then
    echo "📦 Установка Streamlit..."
    pip install streamlit plotly psutil nvidia-ml-py3
fi

# Создание директории для логов если не существует
mkdir -p logs

echo "🌐 Запуск веб-интерфейса..."
echo "Откройте браузер: http://localhost:8501"
echo ""
echo "Для остановки нажмите Ctrl+C"
echo ""

# Запуск Streamlit приложения
streamlit run ira.py --server.port=8501 --server.address=0.0.0.0
