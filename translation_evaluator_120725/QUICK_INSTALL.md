# 🚀 Быстрая установка Translation Evaluator

## Одна команда для полной установки

### Windows (PowerShell/CMD):
```bash
# Универсальная установка (автоматически выберет conda или venv)
setup_complete_env.bat

# Или создание conda окружения из YAML файла
conda env create -f environment.yml

# Или создание venv окружения
python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt
```

### Linux/macOS:
```bash
# Создание conda окружения
conda env create -f environment.yml

# Или создание venv окружения
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

## 📋 Готовые команды для копирования

### Полная установка conda окружения:
```bash
conda env create -f environment.yml
conda activate translation_env
python -c "import streamlit; print('Streamlit:', streamlit.__version__)"
```

### Быстрая установка venv:
```bash
python -m venv translation_env
translation_env\Scripts\activate
pip install -r requirements.txt
```

### Установка с точными версиями:
```bash
conda create -n translation_env python=3.9 -y
conda activate translation_env
pip install -r requirements-exact.txt
```

## 🔧 Проверка установки

```python
# Проверка всех основных пакетов
python -c "
import streamlit; print('✅ Streamlit:', streamlit.__version__)
import torch; print('✅ PyTorch:', torch.__version__)
import transformers; print('✅ Transformers:', transformers.__version__)
import pandas; print('✅ Pandas:', pandas.__version__)
import numpy; print('✅ NumPy:', numpy.__version__)
print('✅ CUDA доступна:', torch.cuda.is_available())
"
```

## 🎯 Минимальные требования

- **Python**: 3.9+
- **RAM**: 8GB (16GB рекомендуется)
- **Диск**: 10GB свободного места
- **OS**: Windows 10+, Linux, macOS

## 📦 Что будет установлено

### Основные пакеты:
- **Streamlit 1.28.0** - веб-интерфейс
- **PyTorch 2.7.1** - машинное обучение
- **Transformers 4.53.1** - NLP модели
- **Pandas 1.5.3** - обработка данных
- **NumPy 1.26.4** - вычисления

### Дополнительные пакеты:
- **SacreBLEU 2.3.1** - метрики оценки
- **python-docx 0.8.11** - обработка Word файлов
- **matplotlib/seaborn** - визуализация
- **scikit-learn 1.2.2** - машинное обучение
- **plotly 5.15.0** - интерактивные графики

## 🚀 Быстрый старт после установки

```bash
# Активация окружения
conda activate translation_env  # или venv\Scripts\activate

# Запуск приложения
streamlit run app.py

# Или через скрипты
start.bat  # главное меню
run_app_conda_fast.bat  # быстрый запуск
```

## 🔧 Альтернативные способы

### Только CPU версия (быстрее):
```bash
conda create -n translation_env python=3.9 -y
conda activate translation_env
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
pip install streamlit transformers pandas numpy python-docx
```

### С GPU поддержкой:
```bash
conda create -n translation_env python=3.9 -y
conda activate translation_env
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

## 📁 Файлы для установки

- **`environment.yml`** - Полная conda конфигурация
- **`requirements.txt`** - Основные pip зависимости
- **`requirements-exact.txt`** - Точные версии пакетов
- **`setup_complete_env.bat`** - Автоматическая установка

## 🚨 Решение проблем

### Если установка не работает:
```bash
# Очистка и переустановка
conda env remove -n translation_env -y
setup_complete_env.bat

# Или ручная установка
conda create -n translation_env python=3.9 -y
conda activate translation_env
pip install streamlit torch transformers pandas numpy
```

### Если не хватает памяти:
- Закройте ненужные программы
- Используйте CPU версию PyTorch
- Увеличьте виртуальную память

### Если медленно работает:
- Используйте SSD диск
- Установите GPU версию PyTorch
- Увеличьте RAM

Выберите наиболее подходящий для вас способ установки! 🎉
