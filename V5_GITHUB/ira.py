import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import subprocess
import os
import json
import time
import threading
from datetime import datetime
import psutil
from sklearn.model_selection import train_test_split

# Инициализация NVIDIA ML (опционально)
try:
    import nvidia_ml_py3 as nvml
    nvml.nvmlInit()
    NVIDIA_AVAILABLE = True
except ImportError:
    NVIDIA_AVAILABLE = False
    nvml = None
except Exception as e:
    NVIDIA_AVAILABLE = False
    nvml = None

# Инициализация ML библиотек (опционально)
try:
    from transformers import MarianTokenizer, MarianMTModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
    from datasets import Dataset
    from huggingface_hub import login, create_repo
    import torch
    import gc
    ML_LIBRARIES_AVAILABLE = True
except ImportError as e:
    ML_LIBRARIES_AVAILABLE = False
    st.warning(f"ML libraries not available: {e}. Some features may be limited.")

class TrainingMonitor:
    def __init__(self):
        self.training_stats = []
        self.is_monitoring = False
        
    def get_system_stats(self):
        """Получить статистику системы"""
        stats = {
            'timestamp': datetime.now(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024**3),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3)
        }
        
        # GPU статистика
        if NVIDIA_AVAILABLE:
            try:
                gpu_count = nvml.nvmlDeviceGetCount()
                for i in range(gpu_count):
                    handle = nvml.nvmlDeviceGetHandleByIndex(i)
                    memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                    utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
                    temperature = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                    
                    stats[f'gpu_{i}_utilization'] = utilization.gpu
                    stats[f'gpu_{i}_memory_used'] = memory_info.used / (1024**2)  # MB
                    stats[f'gpu_{i}_memory_total'] = memory_info.total / (1024**2)  # MB
                    stats[f'gpu_{i}_memory_percent'] = (memory_info.used / memory_info.total) * 100
                    stats[f'gpu_{i}_temperature'] = temperature
            except Exception as e:
                st.error(f"Ошибка получения GPU статистики: {e}")
        
        return stats
    
    def start_monitoring(self):
        """Запустить мониторинг"""
        self.is_monitoring = True
        self.training_stats = []
        
    def stop_monitoring(self):
        """Остановить мониторинг"""
        self.is_monitoring = False
    
    def collect_stats(self):
        """Собрать статистику"""
        if self.is_monitoring:
            stats = self.get_system_stats()
            self.training_stats.append(stats)
            return stats
        return None

def parse_csv(file_path, src_lang, tgt_lang):
    """Парсинг CSV файла"""
    df = pd.read_csv(file_path)
    if src_lang not in df.columns or tgt_lang not in df.columns:
        raise ValueError(f"CSV должен содержать колонки: '{src_lang}', '{tgt_lang}'")
    return [
        {"translation": {src_lang: row[src_lang], tgt_lang: row[tgt_lang]}}
        for _, row in df.iterrows()
        if isinstance(row[src_lang], str) and isinstance(row[tgt_lang], str)
    ]

def get_available_csv_files():
    """Получить список доступных CSV файлов"""
    csv_files = []
    for file in os.listdir('.'):
        if file.endswith('.csv'):
            try:
                size = os.path.getsize(file) / (1024**2)  # MB
                csv_files.append({'name': file, 'size_mb': size})
            except:
                pass
    return csv_files

def get_training_processes():
    """Получить список процессов обучения"""
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent']):
        try:
            if 'train_marian' in ' '.join(proc.info['cmdline']):
                processes.append(proc.info)
        except:
            pass
    return processes

def create_training_dashboard():
    """Создать дашборд обучения"""
    st.title("🚀 MarianMT Training Dashboard")
    
    # Sidebar для настроек
    with st.sidebar:
        st.header("⚙️ Настройки обучения")
        
        # Выбор CSV файла
        csv_files = get_available_csv_files()
        if csv_files:
            csv_options = [f"{f['name']} ({f['size_mb']:.1f} MB)" for f in csv_files]
            selected_csv = st.selectbox("📄 Выберите CSV файл", csv_options)
            csv_file = csv_files[csv_options.index(selected_csv)]['name']
        else:
            st.error("CSV файлы не найдены!")
            return
        
        # Параметры обучения
        st.subheader("🎯 Параметры модели")
        model_name = st.selectbox(
            "Базовая модель",
            ["Helsinki-NLP/opus-mt-en-ru", "Helsinki-NLP/opus-mt-ru-en"]
        )
        
        epochs = st.slider("Количество эпох", 1, 10, 1)
        batch_size = st.slider("Batch size", 4, 32, 24)
        learning_rate = st.select_slider(
            "Learning rate", 
            options=[1e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4],
            value=5e-5,
            format_func=lambda x: f"{x:.0e}"
        )
        
        st.subheader("🏃 Производительность")
        gradient_accumulation = st.slider("Gradient accumulation steps", 1, 8, 1)
        dataloader_workers = st.slider("DataLoader workers", 1, 32, 16)
        
        st.subheader("💾 Сохранение")
        output_dir = st.text_input("Выходная директория", f"./output_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        save_steps = st.slider("Сохранение каждые N шагов", 1000, 10000, 5000)
        
        # Дополнительные опции
        st.subheader("🔧 Дополнительно")
        use_fp16 = st.checkbox("Использовать FP16", True)
        gradient_checkpointing = st.checkbox("Gradient checkpointing", False)
        warmup_steps = st.slider("Warmup steps", 0, 2000, 500)
        
        # Чекбокс для обучения только на валидационном датасете
        train_on_validation = st.checkbox("Обучать только на валидационном датасете", value=False)
    
    # Основной контент
    tab1, tab2, tab3, tab4 = st.tabs(["🎮 Управление", "📊 Мониторинг", "📈 Статистика", "📋 Логи"])
    
    with tab1:
        st.header("🎮 Управление обучением")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🚀 Запуск обучения")
            
            # Проверка токена HuggingFace
            hf_token = st.text_input("HuggingFace Token", type="password")
            repo_name = st.text_input("Название репозитория", f"fine-tuned-marian-{datetime.now().strftime('%Y%m%d')}")
            
            if st.button("🚀 Запустить обучение", type="primary"):
                if not hf_token:
                    st.error("Введите HuggingFace токен!")
                else:
                    # Формирование команды
                    command = f"""
torchrun --nproc_per_node=2 --master_port=29503 train_marian_max_speed.py \\
  --csv {csv_file} \\
  --output_dir {output_dir} \\
  --epochs {epochs} \\
  --batch_size {batch_size} \\
  --gradient_accumulation_steps {gradient_accumulation} \\
  --learning_rate {learning_rate} \\
  --warmup_steps {warmup_steps} \\
  --dataloader_num_workers {dataloader_workers} \\
  --save_steps {save_steps} \\
  --logging_steps 100 \\
  --max_grad_norm 1.0 \\
  --weight_decay 0.01 \\
  --ddp_bucket_cap_mb 100 \\
  --save_total_limit 2
                    """.strip()
                    
                    # Добавление аргумента --train_on_validation, если чекбокс выбран
                    if train_on_validation:
                        command += " \\n  --train_on_validation"
                    
                    st.code(command, language="bash")
                    st.success("Команда сформирована! Запустите в терминале.")
        
        with col2:
            st.subheader("📊 Текущие процессы")
            processes = get_training_processes()
            
            if processes:
                for proc in processes:
                    st.info(f"🔄 PID: {proc['pid']}, CPU: {proc['cpu_percent']:.1f}%, RAM: {proc['memory_percent']:.1f}%")
            else:
                st.warning("Процессы обучения не найдены")
    
    with tab2:
        st.header("📊 Мониторинг в реальном времени")
        
        # Инициализация монитора
        if 'monitor' not in st.session_state:
            st.session_state.monitor = TrainingMonitor()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("▶️ Начать мониторинг"):
                st.session_state.monitor.start_monitoring()
                st.success("Мониторинг запущен!")
        
        with col2:
            if st.button("⏹️ Остановить мониторинг"):
                st.session_state.monitor.stop_monitoring()
                st.info("Мониторинг остановлен")
        
        with col3:
            auto_refresh = st.checkbox("🔄 Автообновление (5с)")
        
        # Автообновление
        if auto_refresh and st.session_state.monitor.is_monitoring:
            time.sleep(5)
            st.rerun()
        
        # Текущая статистика
        if st.session_state.monitor.is_monitoring:
            stats = st.session_state.monitor.collect_stats()
            if stats:
                # Системные метрики
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("CPU", f"{stats['cpu_percent']:.1f}%")
                
                with col2:
                    st.metric("RAM", f"{stats['memory_percent']:.1f}%", 
                             f"{stats['memory_used_gb']:.1f}/{stats['memory_total_gb']:.1f} GB")
                
                # GPU метрики
                if NVIDIA_AVAILABLE:
                    gpu_keys = [k for k in stats.keys() if k.startswith('gpu_') and k.endswith('_utilization')]
                    for i, key in enumerate(gpu_keys):
                        gpu_id = key.split('_')[1]
                        with col3 if i == 0 else col4:
                            st.metric(f"GPU {gpu_id}", f"{stats[key]:.1f}%",
                                     f"{stats[f'gpu_{gpu_id}_temperature']}°C")
    
    with tab3:
        st.header("📈 Статистика производительности")
        
        if st.session_state.monitor.training_stats:
            df_stats = pd.DataFrame(st.session_state.monitor.training_stats)
            
            # График утилизации
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('CPU Utilization', 'Memory Usage', 'GPU Utilization', 'GPU Memory'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # CPU
            fig.add_trace(
                go.Scatter(x=df_stats['timestamp'], y=df_stats['cpu_percent'], name='CPU %'),
                row=1, col=1
            )
            
            # Memory
            fig.add_trace(
                go.Scatter(x=df_stats['timestamp'], y=df_stats['memory_percent'], name='RAM %'),
                row=1, col=2
            )
            
            # GPU (если доступен)
            if NVIDIA_AVAILABLE and 'gpu_0_utilization' in df_stats.columns:
                fig.add_trace(
                    go.Scatter(x=df_stats['timestamp'], y=df_stats['gpu_0_utilization'], name='GPU 0 %'),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=df_stats['timestamp'], y=df_stats['gpu_0_memory_percent'], name='GPU 0 Mem %'),
                    row=2, col=2
                )
            
            fig.update_layout(height=600, title_text="Мониторинг производительности")
            st.plotly_chart(fig, use_container_width=True)
            
            # Сводная таблица
            st.subheader("📊 Статистика")
            st.dataframe(df_stats.describe())
        else:
            st.info("Начните мониторинг для сбора статистики")
    
    with tab4:
        st.header("📋 Логи обучения")
        
        # Поиск лог файлов
        log_files = []
        if os.path.exists('logs'):
            for file in os.listdir('logs'):
                if file.endswith('.log'):
                    log_files.append(file)
        
        if log_files:
            selected_log = st.selectbox("Выберите лог файл", log_files)
            
            if st.button("🔄 Обновить логи"):
                st.rerun()
            
            try:
                with open(f'logs/{selected_log}', 'r', encoding='utf-8') as f:
                    log_content = f.read()
                    
                # Показать последние N строк
                lines = log_content.split('\n')
                num_lines = st.slider("Количество последних строк", 10, 200, 50)
                
                st.text_area("Логи", '\n'.join(lines[-num_lines:]), height=400)
                
                # Парсинг метрик из логов
                loss_lines = [line for line in lines if "'loss':" in line]
                if loss_lines:
                    st.subheader("📉 График Loss")
                    losses = []
                    steps = []
                    
                    for i, line in enumerate(loss_lines[-100:]):  # Последние 100 записей
                        try:
                            # Простое извлечение loss
                            loss_start = line.find("'loss': ") + 8
                            loss_end = line.find(",", loss_start)
                            if loss_end == -1:
                                loss_end = line.find("}", loss_start)
                            
                            loss_value = float(line[loss_start:loss_end])
                            losses.append(loss_value)
                            steps.append(i)
                        except:
                            continue
                    
                    if losses:
                        fig = px.line(x=steps, y=losses, title="Training Loss")
                        fig.update_xaxes(title="Step")
                        fig.update_yaxes(title="Loss")
                        st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Ошибка чтения лога: {e}")
        else:
            st.info("Лог файлы не найдены")

# Основная функция для обучения (оригинальная логика)
def main(args):
    if not ML_LIBRARIES_AVAILABLE:
        st.error("ML libraries (transformers, datasets, torch) are not available. Please install them to use training functionality.")
        return
    
    login(token=args.token)
    src_lang, tgt_lang = ("en", "ru") if "en-ru" in args.model_name else ("ru", "en")
    
    # Загрузка и разбиение на train и validation
    translations = parse_csv(args.csv, src_lang, tgt_lang)
    train_data, val_data = train_test_split(translations, test_size=0.1, random_state=42)

    # Если выбран режим обучения на валидационном датасете
    if hasattr(args, 'train_on_validation') and args.train_on_validation:
        train_data = val_data.copy()
        st.warning("ВНИМАНИЕ: Обучение будет происходить только на валидационном датасете!")
    
    # Преобразуем в Hugging Face Dataset
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    tokenizer = MarianTokenizer.from_pretrained(args.model_name)

    def preprocess_function(examples):
        inputs = [ex[src_lang] for ex in examples['translation']]
        targets = [ex[tgt_lang] for ex in examples['translation']]
        model_inputs = tokenizer(inputs, max_length=args.max_length, truncation=True, padding="max_length")
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=args.max_length, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train = train_dataset.map(
        preprocess_function,
        batched=True,
        batch_size=1000,
        remove_columns=train_dataset.column_names,
        load_from_cache_file=False
    )
    tokenized_val = val_dataset.map(
        preprocess_function,
        batched=True,
        batch_size=1000,
        remove_columns=val_dataset.column_names,
        load_from_cache_file=False
    )

    del train_dataset, val_dataset
    gc.collect()

    model = MarianMTModel.from_pretrained(args.model_name)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        num_train_epochs=args.epochs,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=100,
        fp16=args.fp16 and torch.cuda.is_available(),
        push_to_hub=True,
        hub_model_id=args.repo_name,
        hub_token=args.token,
        report_to="none",
        save_safetensors=True,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=True,
        optim="adamw_torch",
        gradient_checkpointing=args.gradient_checkpointing,
        logging_dir="./logs",
        predict_with_generate=False
    )

    try:
        create_repo(args.repo_name, token=args.token, private=True, exist_ok=True)
    except Exception:
        pass

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    train_result = trainer.train()
    metrics = train_result.metrics
    print(f"Обучение завершено! Final loss: {metrics['train_loss']:.4f}")
    trainer.save_model("final_model")
    trainer.push_to_hub(commit_message="Обучение завершено")

# Точка входа
if __name__ == "__main__":
    # Проверяем, запущен ли как Streamlit приложение
    try:
        # Если запущено через streamlit run, то создаем дашборд
        if 'streamlit' in str(st.__file__):
            create_training_dashboard()
    except:
        # Если запущено как обычный скрипт, то используем argparse
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--csv", type=str, required=True)
        parser.add_argument("--token", type=str, required=True)
        parser.add_argument("--model_name", type=str, default="Helsinki-NLP/opus-mt-en-ru")
        parser.add_argument("--repo_name", type=str, default="fine-tuned-marian")
        parser.add_argument("--epochs", type=int, default=1)
        parser.add_argument("--batch_size", type=int, default=4)
        parser.add_argument("--max_length", type=int, default=128)
        parser.add_argument("--fp16", action="store_true")
        parser.add_argument("--gradient_checkpointing", action="store_true")
        parser.add_argument("--output_dir", type=str, default="./output")
        parser.add_argument("--dataloader_num_workers", type=int, default=2)
        parser.add_argument("--train_on_validation", action="store_true", help="Обучать только на валидационном датасете")
        
        args = parser.parse_args()
        main(args)