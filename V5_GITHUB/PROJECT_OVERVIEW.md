# Проект: Обучение и мониторинг MarianMT на нескольких GPU

## Описание

Данный проект предназначен для обучения нейронных переводческих моделей MarianMT с помощью библиотеки HuggingFace Transformers на одном или нескольких GPU. В проекте реализованы:
- Скрипты для обучения с поддержкой multi-GPU (torchrun)
- Streamlit-дэшборд для управления, мониторинга ресурсов и визуализации логов/метрик
- Автоматическое логирование процесса обучения и построение графиков loss
- Гибкая настройка параметров обучения через CLI и веб-интерфейс

## Основные файлы

- **ira.py** — основной скрипт с веб-интерфейсом (Streamlit dashboard), поддерживает запуск как через Streamlit, так и как обычный Python-скрипт для обучения.
- **train_marian_multi_gpu.py** — скрипт для обучения MarianMT с поддержкой multi-GPU, логированием и автоматическим построением графика loss.
- **requirements.txt** — базовые зависимости для обучения (transformers, datasets, torch и др.)
- **requirements_streamlit.txt** — зависимости для работы Streamlit-дэшборда (streamlit, plotly, psutil, nvidia-ml-py3 и др.)
- **start_dashboard.sh** — скрипт для быстрого запуска Streamlit-дэшборда.
- **STREAMLIT_DASHBOARD.md** — подробная инструкция по использованию веб-интерфейса.
- **cleaned_corpus.csv** и другие .csv — ваши датасеты для обучения/валидации.

## Установка и настройка окружения

1. **Создайте и активируйте виртуальное окружение (рекомендуется):**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **Установите зависимости для обучения:**
   ```bash
   pip install -r requirements.txt
   ```
   Для Streamlit-дэшборда:
   ```bash
   pip install -r requirements_streamlit.txt
   ```

3. **(Опционально) Установите дополнительные утилиты:**
   - Для GPU-мониторинга: `nvidia-ml-py3`
   - Для визуализации: `matplotlib`, `plotly`

## Запуск обучения

### Через CLI (multi-GPU)
```bash
python train_marian_multi_gpu.py \
  --csv your_data.csv \
  --output_dir output \
  --epochs 3 \
  --batch_size 16 \
  --repo_name your_hf_repo \
  --token your_hf_token
```

или через torchrun:
```bash
torchrun --nproc_per_node=2 --master_port=29503 train_marian_multi_gpu.py \
  --csv your_data.csv \
  --output_dir output \
  --epochs 3 \
  --batch_size 16 \
  --repo_name your_hf_repo \
  --token your_hf_token
```

### Через Streamlit-дэшборд
```bash
./start_dashboard.sh
```
или
```bash
streamlit run ira.py
```

## Визуализация логов и метрик
- Все логи обучения пишутся в файл `train.log` (или аналогичный).
- После завершения обучения автоматически строится график loss (`loss_plot.png`).
- В дэшборде доступны вкладки для мониторинга ресурсов, логов и графиков.

## Пример структуры проекта
```
V5/
├── ira.py
├── train_marian_multi_gpu.py
├── requirements.txt
├── requirements_streamlit.txt
├── start_dashboard.sh
├── STREAMLIT_DASHBOARD.md
├── cleaned_corpus.csv
├── ... (другие .csv)
├── output/
│   └── ... (модели, чекпоинты)
├── train.log
├── loss_plot.png
└── ...
```

## Примечания
- Для работы с HuggingFace Hub требуется токен (`--token`).
- Для multi-GPU требуется установленный PyTorch с поддержкой CUDA и корректная настройка NCCL.
- Все параметры обучения можно гибко настраивать через CLI или веб-интерфейс.

---
Если нужна быстрая инструкция или помощь с запуском — смотрите STREAMLIT_DASHBOARD.md или обращайтесь!
