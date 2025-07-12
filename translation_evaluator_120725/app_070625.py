# pylint: disable=line-too-long,too-many-locals,too-many-statements

# Конфигурация Streamlit должна быть первой
import streamlit as st

# Устанавливаем конфигурацию страницы
st.set_page_config(
    page_title="Сравнение переводов моделей и переводчиков",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
import sys
import multiprocessing
import logging
import time
from tqdm import tqdm
from docx import Document
import gc
import xml.etree.ElementTree as ET
import hashlib
from difflib import SequenceMatcher
import asyncio
import signal
import platform
import nest_asyncio
import warnings
import io
import csv
import datetime
from fpdf import FPDF
import zipfile
import re
import traceback
import pandas as pd
import tempfile
import seaborn as sns
import matplotlib.pyplot as plt
import codecs
import torch
from translation_utils.model_loader import load_model_from_zip
from translation_utils.translator import translate_texts
from translation_utils.metrics import compute_metrics_batch
from translation_utils.docx_utils import extract_content_from_docx
import docx
from langdetect import detect

# Добавляем путь к корневой директории проекта в sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Импортируем остальные модули
from translation_utils.tmx_utils import create_tmx_file, TmxCreator, TmxEntry

# Отключаем предупреждения
warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def setup_async_env():
    """Настраивает асинхронное окружение"""
    try:
        # Проверяем текущий цикл событий
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Применяем патч для вложенных циклов событий
        try:
            nest_asyncio.apply()
        except Exception as e:
            logger.warning(f"Не удалось применить nest_asyncio: {str(e)}")
        
        return loop
    except Exception as e:
        logger.error(f"Ошибка при настройке асинхронного окружения: {str(e)}")
        return None

# Настраиваем асинхронное окружение при запуске
setup_async_env()

# Вспомогательные функции
def calculate_file_hash(file_content):
    """Вычисляет хеш содержимого файла"""
    if file_content is None:
        return None
    return hashlib.md5(file_content).hexdigest()

@st.cache_data(ttl=3600)
def get_file_content(uploaded_file):
    """Кэшированное получение содержимого файла"""
    if uploaded_file is None:
        return None
    try:
        return uploaded_file.getvalue()
    except Exception as e:
        logger.error(f"Ошибка при чтении файла: {e}")
        return None

@st.cache_resource(ttl=3600)
def load_cached_model(model_path, extract_path):
    """Кэшированная загрузка модели"""
    try:
        logger.info(f"Начало загрузки модели из {model_path}")
        model, tokenizer = load_model_from_zip(model_path, extract_path)
        logger.info("Модель успешно загружена")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {e}")
        logger.error(traceback.format_exc())
        return None, None

def process_uploaded_file(file):
    """Обработка загруженного файла"""
    try:
        if file.type == "text/csv":
            df = pd.read_csv(file)
            if 'text' not in df.columns:
                st.error("CSV файл должен содержать колонку 'text'")
                return None
            return df['text'].tolist()
        else:
            result = extract_content_from_docx(file)
            return result['sentences'] if result else None
    except Exception as e:
        st.error(f"Ошибка при обработке файла: {str(e)}")
        return None

def cleanup_cuda():
    """Очищает CUDA память"""
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            logger.error(f"Ошибка при очистке CUDA памяти: {str(e)}")

def safe_division(x, y, default=0.0):
    """Безопасное деление с обработкой деления на ноль"""
    try:
        return x / y if y != 0 else default
    except:
        return default

def clean_text_for_export(text):
    """Очищает текст от escape-последовательностей и нормализует для экспорта"""
    if not isinstance(text, str):
        return str(text)
    
    try:
        text = bytes(text, 'utf-8').decode('unicode_escape')
    except:
        pass
    
    text = text.replace('\\r', '').replace('\\n', ' ')
    text = text.replace('\\t', ' ').replace('\\xa0', ' ')
    text = ' '.join(text.split())
    
    return text

# Инициализация состояния сессии
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.update({
        'model1': None,
        'model2': None,
        'tokenizer1': None,
        'tokenizer2': None,
        'source_texts': [],
        'reference_texts': [],
        'translations1': [],
        'translations2': [],
        'yandex_translations': [],
        'google_translations': [],
        'deepl_translations': [],
        'metrics': {},
        'corpus_metrics': {},
        'error': None,
        'session_id': hashlib.md5(str(time.time()).encode()).hexdigest()
    })

def init_streamlit():
    """Инициализация Streamlit"""
    try:
        # Скрываем стандартные стили Streamlit
        hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
        """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)
        
        return True
    except Exception as e:
        logger.error(f"Ошибка при инициализации Streamlit: {str(e)}")
        return False

# Инициализируем Streamlit при запуске
init_streamlit()

# Отключаем предупреждения Streamlit
st.set_option('client.showErrorDetails', False)
st.set_option('client.toolbarMode', 'minimal')

def extract_content_from_csv(file_content) -> dict:
    """Извлекает содержимое из CSV файла"""
    result = {'sentences': [], 'references': [], 'tables': []}

    try:
        # Читаем CSV файл
        csv_data = pd.read_csv(io.BytesIO(file_content))
        
        # Проверяем наличие обязательных столбцов
        required_columns = ['source', 'target']
        if not all(col in csv_data.columns for col in required_columns):
            raise ValueError("CSV файл должен содержать столбцы 'source' и 'target'")
        
        # Извлекаем предложения и референсы
        result['sentences'] = csv_data['source'].tolist()
        result['references'] = csv_data['target'].tolist()
        
    except Exception as e:
        logger.error(f"Ошибка при обработке CSV файла: {e}")
        
    return result

def export_to_csv(data: list, filename: str) -> str | None:
    """Экспортирует данные в CSV файл"""
    try:
        with io.StringIO() as buffer:
            writer = csv.writer(buffer)
            writer.writerows(data)
            return buffer.getvalue()
    except Exception as e:
        logger.error(f"Ошибка при экспорте в CSV: {e}")
        return None

def export_to_pdf(data: list, filename: str) -> bytes | None:
    """Экспортирует данные в PDF файл"""
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.add_font('DejaVu', '', 'DejaVuSansCondensed.ttf', uni=True)
        pdf.set_font('DejaVu', '', 14)
        
        # Добавляем данные
        for row in data:
            pdf.multi_cell(0, 10, str(row))
            pdf.ln()
            
        return pdf.output(dest='S').encode('latin1')
    except Exception as e:
        logger.error(f"Ошибка при экспорте в PDF: {e}")
        return None

def export_to_docx(data, filename) -> bytes | None:
    """Экспортирует данные в DOCX файл"""
    try:
        doc = Document()
        
        # Добавляем данные
        for row in data:
            doc.add_paragraph(str(row))
            
        # Сохраняем во временный буфер
        buffer = io.BytesIO()
        doc.save(buffer)
        return buffer.getvalue()
    except Exception as e:
        logger.error(f"Ошибка при экспорте в DOCX: {e}")
        return None

def export_to_tmx(source_texts, target_texts, filename) -> str | None:
    """Экспортирует параллельные тексты в TMX файл"""
    try:
        tmx_creator = TmxCreator()
        
        # Добавляем записи в TMX
        for src, tgt in zip(source_texts, target_texts):
            entry = TmxEntry(src.strip(), tgt.strip())
            tmx_creator.add_entry(entry)
            
        # Создаем TMX файл
        return tmx_creator.create_tmx_content()
    except Exception as e:
        logger.error(f"Ошибка при экспорте в TMX: {e}")
        return None

def create_metrics_plot(metrics_data) -> bytes | None:
    """Создает визуализацию метрик"""
    try:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=pd.DataFrame(metrics_data))
        plt.title('Distribution of Translation Metrics')
        plt.xticks(rotation=45)
        
        # Сохраняем график во временный буфер
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        plt.close()
        return buffer.getvalue()
    except Exception as e:
        logger.error(f"Ошибка при создании графика: {e}")
        return None

def create_comparison_plot(metrics_data) -> io.BytesIO | None:
    """Создание графика сравнения метрик"""
    try:
        plt.figure(figsize=(12, 6))
        df = pd.DataFrame(metrics_data).round(4)
        df.plot(kind='bar')
        plt.title('Сравнение систем машинного перевода')
        plt.xlabel('Метрика')
        plt.ylabel('Значение')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        return buf
    except Exception as e:
        st.error(f"Ошибка при создании графика: {str(e)}")
        return None

def main():
    """Основная функция приложения"""
    st.title("\U0001F4C4 Сравнение переводов моделей и машинных переводчиков")
    
    # Добавляем возможность загрузки CSV
    uploaded_docx = st.file_uploader("\U0001F4E5 Загрузите исходный документ (Word или CSV)", type=["docx", "csv"])
    uploaded_ref = st.file_uploader("\U0001F4E5 Загрузите эталонный перевод (опционально)", type=["docx", "csv"])

    model1_zip = st.file_uploader("\U0001F9E0 Модель 1 (дообученная)", type=["zip"])
    model2_zip = st.file_uploader("\U0001F9E0 Модель 2 (исходная)", type=["zip"])

    yandex_docx = st.file_uploader("\U0001F916 Перевод Яндекс (DOCX или CSV, опционально)", type=["docx", "csv"])
    google_docx = st.file_uploader("\U0001F916 Перевод Google (DOCX или CSV, опционально)", type=["docx", "csv"])
    deepl_docx = st.file_uploader("\U0001F916 Перевод DeepL (DOCX или CSV, опционально)", type=["docx", "csv"])

    model1_name = st.text_input("Название Модели 1", value="Model 1")
    model2_name = st.text_input("Название Модели 2", value="Model 2")

    # Обработка загруженных файлов
    if uploaded_docx:
        source_texts = process_uploaded_file(uploaded_docx)
        if source_texts:
            st.session_state.source_texts = source_texts
            st.write(f"Загружено исходных предложений: {len(source_texts)}")

    if uploaded_ref:
        reference_texts = process_uploaded_file(uploaded_ref)
        if reference_texts:
            st.session_state.reference_texts = reference_texts
            st.write(f"Загружено эталонных переводов: {len(reference_texts)}")

    # Загрузка моделей
    if model1_zip and not st.session_state.model1:
        st.session_state.model1, st.session_state.tokenizer1 = load_cached_model(model1_zip, None)

    if model2_zip and not st.session_state.model2:
        st.session_state.model2, st.session_state.tokenizer2 = load_cached_model(model2_zip, None)

    # Обработка переводов от других систем
    if yandex_docx:
        st.session_state.yandex_translations = process_uploaded_file(yandex_docx)
        
    if google_docx:
        st.session_state.google_translations = process_uploaded_file(google_docx)
        
    if deepl_docx:
        st.session_state.deepl_translations = process_uploaded_file(deepl_docx)

    # Кнопка для запуска перевода и оценки
    col1, col2 = st.columns([1, 3])
    with col1:
        start_button = st.button("🚀 Запустить", use_container_width=True)
    with col2:
        st.markdown("Нажмите для запуска перевода и оценки")

    if start_button:
        if not st.session_state.source_texts:
            st.error("Загрузите исходный документ!")
        else:
            try:
                # Перевод с помощью Модели 1
                if st.session_state.model1 and st.session_state.tokenizer1:
                    with st.spinner(f"Перевод с помощью {model1_name}..."):
                        translations1 = translate_texts(
                            st.session_state.source_texts,
                            st.session_state.model1,
                            st.session_state.tokenizer1
                        )
                        st.session_state.translations1 = translations1
                        
                        if st.session_state.reference_texts:
                            metrics1 = compute_metrics_batch(translations1, st.session_state.reference_texts)
                            st.session_state.metrics[model1_name] = {k: sum(v)/len(v) for k, v in metrics1.items()}
                
                # Перевод с помощью Модели 2
                if st.session_state.model2 and st.session_state.tokenizer2:
                    with st.spinner(f"Перевод с помощью {model2_name}..."):
                        translations2 = translate_texts(
                            st.session_state.source_texts,
                            st.session_state.model2,
                            st.session_state.tokenizer2
                        )
                        st.session_state.translations2 = translations2
                        
                        if st.session_state.reference_texts:
                            metrics2 = compute_metrics_batch(translations2, st.session_state.reference_texts)
                            st.session_state.metrics[model2_name] = {k: sum(v)/len(v) for k, v in metrics2.items()}
                
                # Оценка переводов от других систем
                if st.session_state.reference_texts:
                    if st.session_state.yandex_translations:
                        metrics_yandex = compute_metrics_batch(
                            st.session_state.yandex_translations,
                            st.session_state.reference_texts
                        )
                        st.session_state.metrics['Yandex'] = {k: sum(v)/len(v) for k, v in metrics_yandex.items()}
                    
                    if st.session_state.google_translations:
                        metrics_google = compute_metrics_batch(
                            st.session_state.google_translations,
                            st.session_state.reference_texts
                        )
                        st.session_state.metrics['Google'] = {k: sum(v)/len(v) for k, v in metrics_google.items()}
                    
                    if st.session_state.deepl_translations:
                        metrics_deepl = compute_metrics_batch(
                            st.session_state.deepl_translations,
                            st.session_state.reference_texts
                        )
                        st.session_state.metrics['DeepL'] = {k: sum(v)/len(v) for k, v in metrics_deepl.items()}
                
                st.success("✅ Перевод и оценка выполнены!")
                
            except Exception as e:
                st.error(f"❌ Ошибка при переводе и оценке: {str(e)}")

    # Отображение результатов
    if st.session_state.metrics:
        st.subheader("📊 Результаты оценки")
        
        # Создаем DataFrame для сравнения
        comparison_df = pd.DataFrame(st.session_state.metrics).round(4)
        st.table(comparison_df)
        
        # Создаем график сравнения
        plot_buf = create_comparison_plot(st.session_state.metrics)
        if plot_buf:
            st.image(plot_buf)

    # Отображение переводов
    if st.session_state.source_texts:
        st.subheader("📝 Сравнение переводов")
        
        # Создаем DataFrame с переводами
        translations_data = {}
        
        # Определяем язык исходного текста для каждого предложения
        for i, text in enumerate(st.session_state.source_texts):
            try:
                lang = detect(text)
                if lang == 'en':
                    translations_data['Исходный текст (en)'] = st.session_state.source_texts
                elif lang == 'ru':
                    translations_data['Исходный текст (ru)'] = st.session_state.source_texts
                else:
                    translations_data['Исходный текст'] = st.session_state.source_texts
                break
            except:
                translations_data['Исходный текст'] = st.session_state.source_texts
                break
        
        if st.session_state.reference_texts:
            translations_data['Эталонный перевод (из файла)'] = st.session_state.reference_texts
        
        if st.session_state.translations1:
            translations_data[f"{model1_name} (из файла)"] = st.session_state.translations1
        if st.session_state.translations2:
            translations_data[f"{model2_name} (из файла)"] = st.session_state.translations2
        if st.session_state.yandex_translations:
            translations_data['Yandex (API)'] = st.session_state.yandex_translations
        if st.session_state.google_translations:
            translations_data['Google (API)'] = st.session_state.google_translations
        if st.session_state.deepl_translations:
            translations_data['DeepL (API)'] = st.session_state.deepl_translations
        
        translations_df = pd.DataFrame(translations_data)
        st.dataframe(translations_df)

if __name__ == "__main__":
    main()