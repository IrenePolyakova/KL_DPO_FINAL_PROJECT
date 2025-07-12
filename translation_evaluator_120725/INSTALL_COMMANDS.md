# 📋 Translation Evaluator - Основные команды установки

## 🚀 Быстрая установка (Рекомендуется)

### Одна команда для всего:
```bash
setup_complete_env.bat
```

## 🐍 Conda установка

### Из готового файла:
```bash
conda env create -f environment.yml
conda activate translation_env
```

### Полная команда:
```bash
conda create -n translation_env python=3.9 -y && conda activate translation_env && conda install pytorch torchvision torchaudio cpuonly -c pytorch -y && pip install streamlit>=1.28.0 transformers>=4.30.0 pandas>=1.5.3 numpy>=1.24.3 python-docx>=0.8.11 sacrebleu>=2.3.1 matplotlib>=3.7.1 seaborn>=0.12.2 scikit-learn>=1.2.2 plotly>=5.15.0 openpyxl>=3.0.0 tqdm>=4.65.0 safetensors>=0.3.1 datasets>=2.10.0 nltk>=3.8.0
```

## 🔧 Venv установка

### Полная команда:
```bash
python -m venv translation_env && translation_env\Scripts\activate && pip install -r requirements.txt
```

## 🎯 Минимальная установка

### Только основные пакеты:
```bash
pip install streamlit torch transformers pandas numpy python-docx sacrebleu matplotlib plotly
```

## 📦 Проверка установки

```python
python -c "import streamlit; import torch; import transformers; import pandas; import numpy; print('✅ Все пакеты установлены успешно!')"
```

## 🚀 Запуск после установки

```bash
# Активация окружения
conda activate translation_env  # или translation_env\Scripts\activate

# Запуск приложения
streamlit run app.py

# Или через скрипты
start.bat
```

## 📋 Список основных пакетов

- **streamlit** - веб-интерфейс
- **torch** - машинное обучение  
- **transformers** - NLP модели
- **pandas** - обработка данных
- **numpy** - вычисления
- **python-docx** - работа с Word
- **sacrebleu** - метрики оценки
- **matplotlib/plotly** - визуализация
- **openpyxl** - работа с Excel

## 🔧 Файлы для установки

- **`environment.yml`** - Conda конфигурация
- **`requirements.txt`** - Pip зависимости
- **`requirements-exact.txt`** - Точные версии
- **`setup_complete_env.bat`** - Автоустановка

Выберите наиболее подходящий способ установки! 🎉
