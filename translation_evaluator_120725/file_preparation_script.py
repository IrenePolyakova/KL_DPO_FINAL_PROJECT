#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для подготовки файлов из билингвального CSV для Translation Evaluator

Входной файл: CSV с колонками en (исходный текст) и ru (эталонный перевод)
Выходные файлы:
- source_only.csv - только исходные тексты (en)
- reference_only.csv - только эталонные переводы (ru)
- bilingual_full.csv - полный билингвальный файл (en + ru)
- translation_template.csv - шаблон для переводов других систем
"""

import pandas as pd
import os
import sys
import argparse
from pathlib import Path
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_xlsx_to_csv(xlsx_file, output_dir=None):
    """
    Конвертирует XLSX файл в CSV UTF-8
    
    Args:
        xlsx_file: Путь к XLSX файлу
        output_dir: Директория для сохранения (по умолчанию рядом с исходным)
    
    Returns:
        str: Путь к созданному CSV файлу
    """
    if output_dir is None:
        output_dir = os.path.dirname(xlsx_file)
    
    # Определяем имя выходного файла
    xlsx_name = Path(xlsx_file).stem
    csv_file = os.path.join(output_dir, f"{xlsx_name}.csv")
    
    try:
        logger.info(f"Конвертация XLSX в CSV: {xlsx_file} -> {csv_file}")
        
        # Читаем Excel файл
        df = pd.read_excel(xlsx_file)
        logger.info(f"Загружено {len(df)} строк из XLSX")
        
        # Сохраняем как CSV с кодировкой UTF-8
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        logger.info(f"Файл сохранен как CSV: {csv_file}")
        
        return csv_file
        
    except Exception as e:
        logger.error(f"Ошибка при конвертации XLSX: {str(e)}")
        raise

def detect_columns(df):
    """Автоматически определяет колонки с английским и русским текстом"""
    columns_lower = {col.lower(): col for col in df.columns}
    
    src_col = None
    tgt_col = None
    
    # Ищем колонки с английским текстом
    for col in df.columns:
        col_lower = col.lower()
        if not src_col and any(x in col_lower for x in ['source', 'src', 'en', 'исходный', 'english']):
            src_col = col
            break
    
    # Ищем колонки с русским текстом/переводами
    for col in df.columns:
        col_lower = col.lower()
        if not tgt_col and any(x in col_lower for x in ['target', 'tgt', 'ru', 'эталон', 'перевод', 'russian', 'reference']):
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

def prepare_files(input_file, output_dir=None, prefix="", filter_empty=True):
    """
    Подготавливает файлы из билингвального CSV или XLSX
    
    Args:
        input_file: Путь к исходному CSV или XLSX файлу
        output_dir: Директория для сохранения файлов (по умолчанию - рядом с исходным)
        prefix: Префикс для имен выходных файлов
        filter_empty: Фильтровать пустые строки и дубли
    """
    
    # Проверяем существование файла
    if not os.path.exists(input_file):
        logger.error(f"Файл не найден: {input_file}")
        return False
    
    # Определяем тип файла и конвертируем XLSX в CSV если нужно
    file_ext = Path(input_file).suffix.lower()
    if file_ext == '.xlsx':
        logger.info("Обнаружен XLSX файл. Конвертируем в CSV...")
        try:
            csv_file = convert_xlsx_to_csv(input_file, output_dir)
            input_file = csv_file  # Используем конвертированный CSV
            logger.info("✅ Конвертация XLSX в CSV завершена")
        except Exception as e:
            logger.error(f"Не удалось конвертировать XLSX в CSV: {str(e)}")
            return False
    elif file_ext != '.csv':
        logger.error(f"Неподдерживаемый формат файла: {file_ext}. Поддерживаются только CSV и XLSX")
        return False
    
    # Определяем директорию для сохранения
    if output_dir is None:
        output_dir = os.path.dirname(input_file)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Определяем префикс из имени файла, если не задан
    if not prefix:
        prefix = Path(input_file).stem
    
    try:
        # Читаем исходный файл
        logger.info(f"Чтение файла: {input_file}")
        df = pd.read_csv(input_file, encoding='utf-8-sig')
        logger.info(f"Загружено {len(df)} строк")
        
        # Определяем колонки
        src_col, tgt_col = detect_columns(df)
        
        if not src_col or not tgt_col:
            logger.error("Не удалось определить колонки с исходным текстом и переводом")
            logger.info(f"Доступные колонки: {list(df.columns)}")
            return False
        
        logger.info(f"Исходный текст: колонка '{src_col}'")
        logger.info(f"Эталонный перевод: колонка '{tgt_col}'")
        
        # Очищаем данные
        df[src_col] = df[src_col].apply(clean_text)
        df[tgt_col] = df[tgt_col].apply(clean_text)
        
        # Фильтруем пустые строки и дубли
        if filter_empty:
            original_len = len(df)
            # Убираем строки где исходный текст или перевод пустые
            df = df[(df[src_col] != "") & (df[tgt_col] != "")]
            # Убираем строки где перевод совпадает с исходным текстом
            df = df[df[src_col] != df[tgt_col]]
            # Убираем дубли
            df = df.drop_duplicates(subset=[src_col])
            logger.info(f"После фильтрации: {len(df)} строк (удалено {original_len - len(df)})")
        
        # Создаем индекс для удобства
        df = df.reset_index(drop=True)
        df.index += 1
        
        # 1. Файл только с исходными текстами (для загрузки как source в app.py)
        source_only = pd.DataFrame({
            'en': df[src_col]
        })
        source_file = os.path.join(output_dir, f"{prefix}_source_only.csv")
        source_only.to_csv(source_file, index=False, encoding='utf-8-sig')
        logger.info(f"Создан файл с исходными текстами: {source_file}")
        
        # 2. Файл только с эталонными переводами (для загрузки как reference в app.py)
        reference_only = pd.DataFrame({
            'ru': df[tgt_col]
        })
        reference_file = os.path.join(output_dir, f"{prefix}_reference_only.csv")
        reference_only.to_csv(reference_file, index=False, encoding='utf-8-sig')
        logger.info(f"Создан файл с эталонными переводами: {reference_file}")
        
        # 3. Полный билингвальный файл (для загрузки как source в app.py - содержит и исходные, и эталонные)
        bilingual_full = pd.DataFrame({
            'en': df[src_col],
            'ru': df[tgt_col]
        })
        bilingual_file = os.path.join(output_dir, f"{prefix}_bilingual_full.csv")
        bilingual_full.to_csv(bilingual_file, index=False, encoding='utf-8-sig')
        logger.info(f"Создан полный билингвальный файл: {bilingual_file}")
        
        # 4. Шаблон для переводов других систем (только исходные тексты + пустая колонка для переводов)
        translation_template = pd.DataFrame({
            'en': df[src_col],
            'ru': [''] * len(df)
        })
        template_file = os.path.join(output_dir, f"{prefix}_translation_template.csv")
        translation_template.to_csv(template_file, index=False, encoding='utf-8-sig')
        logger.info(f"Создан шаблон для переводов: {template_file}")
        
        # 5. Файл с ID для удобства отслеживания
        indexed_file = pd.DataFrame({
            'ID': range(1, len(df) + 1),
            'en': df[src_col],
            'ru': df[tgt_col]
        })
        indexed_path = os.path.join(output_dir, f"{prefix}_indexed.csv")
        indexed_file.to_csv(indexed_path, index=False, encoding='utf-8-sig')
        logger.info(f"Создан индексированный файл: {indexed_path}")
        
        # Создаем инструкцию по использованию
        instructions = f"""
# Инструкция по использованию подготовленных файлов

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
- Всего предложений: {len(df)}
- Исходный текст: колонка '{src_col}'
- Эталонный перевод: колонка '{tgt_col}'
- Фильтрация пустых строк: {'включена' if filter_empty else 'отключена'}
"""
        
        instructions_file = os.path.join(output_dir, f"{prefix}_INSTRUCTIONS.md")
        with open(instructions_file, 'w', encoding='utf-8') as f:
            f.write(instructions)
        logger.info(f"Создана инструкция: {instructions_file}")
        
        logger.info("\n" + "="*50)
        logger.info("ПОДГОТОВКА ФАЙЛОВ ЗАВЕРШЕНА УСПЕШНО!")
        logger.info("="*50)
        logger.info(f"Обработано предложений: {len(df)}")
        logger.info(f"Файлы сохранены в: {output_dir}")
        logger.info(f"Префикс файлов: {prefix}")
        
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при обработке файла: {str(e)}")
        return False

def main():
    """Основная функция для запуска из командной строки"""
    parser = argparse.ArgumentParser(description="Подготовка файлов из билингвального CSV/XLSX для Translation Evaluator")
    parser.add_argument("input_file", help="Путь к исходному CSV или XLSX файлу")
    parser.add_argument("-o", "--output", help="Директория для сохранения файлов")
    parser.add_argument("-p", "--prefix", help="Префикс для имен выходных файлов")
    parser.add_argument("--no-filter", action="store_true", help="Не фильтровать пустые строки и дубли")
    
    args = parser.parse_args()
    
    # Проверяем формат файла
    file_ext = Path(args.input_file).suffix.lower()
    if file_ext not in ['.csv', '.xlsx']:
        print(f"\n❌ Неподдерживаемый формат файла: {file_ext}")
        print("📋 Поддерживаются только файлы CSV и XLSX")
        sys.exit(1)
    
    success = prepare_files(
        input_file=args.input_file,
        output_dir=args.output,
        prefix=args.prefix,
        filter_empty=not args.no_filter
    )
    
    if success:
        print("\n✅ Файлы успешно подготовлены!")
        print("📖 Ознакомьтесь с инструкцией в файле *_INSTRUCTIONS.md")
        if Path(args.input_file).suffix.lower() == '.xlsx':
            print("📄 XLSX файл был автоматически конвертирован в CSV UTF-8")
        sys.exit(0)
    else:
        print("\n❌ Ошибка при подготовке файлов")
        sys.exit(1)

if __name__ == "__main__":
    main()
