# Установка Translation Evaluator

## Системные требования

- Python 3.8 или выше
- CUDA-совместимая видеокарта (опционально, для GPU-ускорения)
- 8GB RAM минимум (рекомендуется 16GB+)
- 2GB свободного места на диске

## Установка из исходного кода

1. Клонируйте репозиторий:
```bash
git clone https://github.com/your-username/translation-evaluator.git
cd translation-evaluator
```

2. Создайте виртуальное окружение:
```bash
# С использованием conda (рекомендуется)
conda create -n translation-evaluator python=3.8
conda activate translation-evaluator

# Или с использованием venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Установите зависимости:
```bash
# Основные зависимости
pip install -r requirements.txt

# Зависимости для разработки (опционально)
pip install -e ".[dev]"
```

## Проверка установки

1. Запустите тесты:
```bash
pytest
```

2. Запустите приложение:
```bash
streamlit run app.py
```

## Возможные проблемы

### CUDA не обнаружен

Если PyTorch не видит CUDA:
1. Проверьте установку CUDA и cuDNN
2. Переустановите PyTorch с поддержкой CUDA:
```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Ошибки импорта

Если возникают ошибки импорта:
1. Убедитесь, что вы в корректном виртуальном окружении
2. Проверьте, что все зависимости установлены:
```bash
pip list
```

## Дополнительные компоненты

### Установка GPU драйверов

1. NVIDIA драйверы: [Download](https://www.nvidia.com/Download/index.aspx)
2. CUDA Toolkit: [Download](https://developer.nvidia.com/cuda-downloads)
3. cuDNN: [Download](https://developer.nvidia.com/cudnn)

### Установка системных зависимостей

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install python3-dev build-essential
```

#### Windows
- Visual C++ Build Tools: [Download](https://visualstudio.microsoft.com/visual-cpp-build-tools/) 