import streamlit as st
import pandas as pd
import io
import os
import zipfile
from pathlib import Path
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Настройка страницы
st.set_page_config(
    page_title="Подготовка файлов для Translation Evaluator",
    page_icon="📁",
    layout="wide",
    initial_sidebar_state="expanded"
)


def convert_xlsx_to_csv_data(uploaded_file):
    """
    Конвертирует загруженный XLSX файл в DataFrame
    
    Args:
        uploaded_file: Загруженный XLSX файл из Streamlit
    
    Returns:
        pd.DataFrame: Данные из XLSX файла
    """
    try:
        # Читаем Excel файл
        df = pd.read_excel(uploaded_file)
        logger.info(f"Загружено {len(df)} строк из XLSX")
        return df
    except Exception as e:
        logger.error(f"Ошибка при чтении XLSX: {str(e)}")
        raise

def detect_columns(df):
    """Автоматически определяет колонки с английским и русским текстом"""
    src_col = None
    tgt_col = None
    
    # Ищем колонки с английским текстом
    for col in df.columns:
        col_lower = col.lower()
        if not src_col and any(x in col_lower for x in [
            'source', 'src', 'en', 'исходный', 'english'
        ]):
            src_col = col
            break
    
    # Ищем колонки с русским текстом/переводами
    for col in df.columns:
        col_lower = col.lower()
        if not tgt_col and any(x in col_lower for x in [
            'target', 'tgt', 'ru', 'эталон', 'перевод', 'russian', 'reference'
        ]):
            tgt_col = col
            break
    
    # Если не нашли по ключевым словам, берем первые две колонки
    if not src_col and len(df.columns) >= 2:
        src_col = df.columns[0]
    if not tgt_col and len(df.columns) >= 2:
        tgt_col = df.columns[1]
    
    return src_col, tgt_col

def clean_text(text):
    """Очищает текст от лишних символов"""
    if pd.isna(text):
        return ""
    return str(text).strip()

def prepare_files_data(df, src_col, tgt_col, prefix="prepared", filter_empty=True):
    """
    Подготавливает данные для файлов из билингвального DataFrame
    
    Returns:
        dict: Словарь с подготовленными данными для каждого файла
    """
    
    # Создаем копию DataFrame
    df_clean = df.copy()
    
    # Очищаем данные
    df_clean[src_col] = df_clean[src_col].apply(clean_text)
    df_clean[tgt_col] = df_clean[tgt_col].apply(clean_text)
    
    # Фильтруем пустые строки и дубли
    if filter_empty:
        original_len = len(df_clean)
        # Убираем строки где исходный текст или перевод пустые
        df_clean = df_clean[(df_clean[src_col] != "") & (df_clean[tgt_col] != "")]
        # Убираем строки где перевод совпадает с исходным текстом
        df_clean = df_clean[df_clean[src_col] != df_clean[tgt_col]]
        # Убираем дубли
        df_clean = df_clean.drop_duplicates(subset=[src_col])
        filtered_count = original_len - len(df_clean)
    else:
        filtered_count = 0
    
    # Создаем индекс для удобства
    df_clean = df_clean.reset_index(drop=True)
    
    # Подготавливаем данные для разных файлов
    files_data = {}
    
    # 1. Файл только с исходными текстами
    files_data[f"{prefix}_source_only.csv"] = pd.DataFrame({
        'en': df_clean[src_col]
    })
    
    # 2. Файл только с эталонными переводами
    files_data[f"{prefix}_reference_only.csv"] = pd.DataFrame({
        'ru': df_clean[tgt_col]
    })
    
    # 3. Полный билингвальный файл
    files_data[f"{prefix}_bilingual_full.csv"] = pd.DataFrame({
        'en': df_clean[src_col],
        'ru': df_clean[tgt_col]
    })
    
    # 4. Шаблон для переводов других систем
    files_data[f"{prefix}_translation_template.csv"] = pd.DataFrame({
        'en': df_clean[src_col],
        'ru': [''] * len(df_clean)
    })
    
    # 5. Файл с ID для удобства отслеживания
    files_data[f"{prefix}_indexed.csv"] = pd.DataFrame({
        'ID': range(1, len(df_clean) + 1),
        'en': df_clean[src_col],
        'ru': df_clean[tgt_col]
    })
    
    return files_data, len(df_clean), filtered_count

def create_instructions(prefix, total_sentences, src_col, tgt_col, filter_empty):
    """Создает инструкцию по использованию"""
    instructions = f"""# Инструкция по использованию подготовленных файлов

## Сгенерированные файлы:

1. **{prefix}_bilingual_full.csv** - основной файл для загрузки
   - Содержит исходные тексты (en) и эталонные переводы (ru)
   - Загружайте как "Исходный документ" в Translation Evaluator
   - НЕ загружайте отдельно как "Эталонный перевод"

2. **{prefix}_source_only.csv** - только исходные тексты
   - Используйте если нужно загрузить исходные тексты отдельно
   - Тогда загрузите {prefix}_reference_only.csv как "Эталонный перевод"

3. **{prefix}_reference_only.csv** - только эталонные переводы
   - Используйте вместе с {prefix}_source_only.csv

4. **{prefix}_translation_template.csv** - шаблон для переводов
   - Скопируйте этот файл для каждой системы перевода
   - Заполните колонку 'ru' переводами от Google/Yandex/других систем
   - Загрузите как дополнительные переводы в приложение

5. **{prefix}_indexed.csv** - с ID для отслеживания
   - Удобно для анализа конкретных предложений

## Рекомендуемый способ использования:

### Вариант 1 (проще):
- Исходный документ: {prefix}_bilingual_full.csv
- Эталонный перевод: НЕ загружать
- Переводы других систем: заполненные шаблоны

### Вариант 2 (раздельно):
- Исходный документ: {prefix}_source_only.csv
- Эталонный перевод: {prefix}_reference_only.csv
- Переводы других систем: заполненные шаблоны

## Статистика:
- Всего предложений: {total_sentences}
- Исходный текст: колонка '{src_col}'
- Эталонный перевод: колонка '{tgt_col}'
- Фильтрация пустых строк: {'включена' if filter_empty else 'отключена'}
"""
    return instructions

def main():
    st.title("📁 Подготовка файлов для Translation Evaluator")
    
    st.markdown("""
    Это приложение поможет вам подготовить файлы из билингвального CSV 
    для использования в Translation Evaluator.
    """)
    
    # Боковая панель с инструкциями
    with st.sidebar:
        st.header("📖 Инструкции")
        st.markdown("""
        **Входной файл должен содержать:**
        - Колонку с исходным текстом (en)
        - Колонку с эталонным переводом (ru)
        
        **Поддерживаемые форматы:**
        - CSV (UTF-8, любая кодировка)
        - XLSX (Excel файлы)
        
        **Приложение создаст:**
        - Файл только с исходными текстами
        - Файл только с эталонными переводами  
        - Полный билингвальный файл
        - Шаблон для переводов других систем
        - Индексированный файл
        
        **XLSX файлы:**
        - Автоматически конвертируются в CSV UTF-8
        - Сохраняется исходная структура данных
        """)
    
    # Загрузка файла
    uploaded_file = st.file_uploader(
        "Выберите CSV или XLSX файл с билингвальными данными", 
        type=['csv', 'xlsx']
    )
    
    if uploaded_file is not None:
        try:
            # Определяем тип файла и читаем соответственно
            file_extension = Path(uploaded_file.name).suffix.lower()
            
            if file_extension == '.xlsx':
                st.info("📄 Обнаружен XLSX файл. Конвертируем в CSV UTF-8...")
                df = convert_xlsx_to_csv_data(uploaded_file)
                st.success("✅ XLSX файл успешно конвертирован!")
            elif file_extension == '.csv':
                df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
            else:
                st.error("❌ Неподдерживаемый формат файла")
                return
            
            st.success(f"✅ Файл загружен успешно! Строк: {len(df)}")
            
            if file_extension == '.xlsx':
                st.info("💡 Исходный XLSX файл будет автоматически сохранен как CSV UTF-8")
            
            # Показываем первые строки
            st.subheader("🔍 Предварительный просмотр данных")
            st.dataframe(df.head(), use_container_width=True)
            
            # Автоматически определяем колонки
            src_col_auto, tgt_col_auto = detect_columns(df)
            
            # Настройки
            st.subheader("⚙️ Настройки")
            
            col1, col2 = st.columns(2)
            
            with col1:
                src_col = st.selectbox(
                    "Колонка с исходным текстом (en)",
                    options=df.columns,
                    index=df.columns.get_loc(src_col_auto) if src_col_auto else 0
                )
                
                prefix = st.text_input(
                    "Префикс для имен файлов",
                    value=Path(uploaded_file.name).stem
                )
            
            with col2:
                tgt_col = st.selectbox(
                    "Колонка с эталонным переводом (ru)",
                    options=df.columns,
                    index=df.columns.get_loc(tgt_col_auto) if tgt_col_auto else 1
                )
                
                filter_empty = st.checkbox(
                    "Фильтровать пустые строки и дубли",
                    value=True,
                    help="Убирает строки где перевод пустой или совпадает с исходным текстом"
                )
            
            # Показываем выбранные колонки
            if src_col and tgt_col:
                st.subheader("📋 Выбранные данные")
                preview_df = pd.DataFrame({
                    'Исходный текст': df[src_col].head(3),
                    'Эталонный перевод': df[tgt_col].head(3)
                })
                st.dataframe(preview_df, use_container_width=True)
            
            # Кнопка для подготовки файлов
            if st.button("🚀 Подготовить файлы", type="primary"):
                if src_col and tgt_col:
                    with st.spinner("Подготовка файлов..."):
                        try:
                            # Подготавливаем данные
                            files_data, total_sentences, filtered_count = prepare_files_data(
                                df, src_col, tgt_col, prefix, filter_empty
                            )
                            
                            # Создаем инструкцию
                            instructions = create_instructions(
                                prefix, total_sentences, src_col, tgt_col, filter_empty
                            )
                            
                            # Показываем статистику
                            st.success("✅ Файлы подготовлены успешно!")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Всего предложений", total_sentences)
                            with col2:
                                st.metric("Отфильтровано", filtered_count)
                            with col3:
                                st.metric("Файлов создано", len(files_data))
                            
                            # Создаем ZIP архив для скачивания
                            zip_buffer = io.BytesIO()
                            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                # Добавляем CSV файлы
                                for filename, data in files_data.items():
                                    csv_data = data.to_csv(index=False).encode('utf-8-sig')
                                    zip_file.writestr(filename, csv_data)
                                
                                # Добавляем инструкцию
                                zip_file.writestr(f"{prefix}_INSTRUCTIONS.md", instructions.encode('utf-8'))
                            
                            zip_buffer.seek(0)
                            
                            # Кнопка для скачивания
                            st.download_button(
                                label="📥 Скачать все файлы (ZIP)",
                                data=zip_buffer.getvalue(),
                                file_name=f"{prefix}_prepared_files.zip",
                                mime="application/zip"
                            )
                            
                            # Показываем список файлов
                            st.subheader("📁 Подготовленные файлы")
                            for filename, data in files_data.items():
                                with st.expander(f"📄 {filename} ({len(data)} строк)"):
                                    st.dataframe(data.head(), use_container_width=True)
                                    
                                    # Кнопка для скачивания отдельного файла
                                    csv_data = data.to_csv(index=False).encode('utf-8-sig')
                                    st.download_button(
                                        label=f"📥 Скачать {filename}",
                                        data=csv_data,
                                        file_name=filename,
                                        mime="text/csv",
                                        key=f"download_{filename}"
                                    )
                            
                            # Показываем инструкцию
                            st.subheader("📖 Инструкция по использованию")
                            st.markdown(instructions)
                            
                        except Exception as e:
                            st.error(f"❌ Ошибка при подготовке файлов: {str(e)}")
                            st.exception(e)
                else:
                    st.error("❌ Выберите колонки с исходным текстом и эталонным переводом")
        
        except Exception as e:
            st.error(f"❌ Ошибка при чтении файла: {str(e)}")
            st.info("Убедитесь, что файл имеет правильный формат CSV и кодировку UTF-8")

if __name__ == "__main__":
    main()
