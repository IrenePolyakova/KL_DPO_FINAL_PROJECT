import torch
import logging
from typing import List, Tuple, Optional
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import math
from functools import partial

logger = logging.getLogger(__name__)

def translate_chunk(
    texts: List[str],
    model: torch.nn.Module,
    tokenizer,
    batch_size: int = 8,
    max_length: int = 512,
    num_beams: int = 4,
    device: Optional[str] = None,
    show_progress: bool = False
) -> Tuple[List[str], float]:
    """
    Переводит список текстов с оптимизациями для CPU/GPU
    
    Args:
        texts: Список текстов для перевода
        model: Модель для перевода
        tokenizer: Токенизатор
        batch_size: Размер батча
        max_length: Максимальная длина последовательности
        num_beams: Количество лучей для beam search
        device: Устройство для вычислений ('cuda' или 'cpu')
        show_progress: Показывать ли прогресс-бар
        
    Returns:
        Tuple[List[str], float]: (переводы, скорость перевода в предл./сек)
    """
    if not texts:
        return [], 0.0
    
    # Определяем устройство
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.eval()
    model.to(device)
    
    translations = []
    total_tokens = 0
    start_time = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
    end_time = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
    
    if start_time:
        start_time.record()
    
    try:
        # Создаем итератор с прогресс-баром если нужно
        batches = range(0, len(texts), batch_size)
        if show_progress:
            batches = tqdm(batches, desc="Перевод", unit="batch")
        
        for i in batches:
            batch_texts = texts[i:i + batch_size]
            
            # Создаем маппинг для пустых строк
            empty_indices = []
            non_empty_texts = []
            for idx, text in enumerate(batch_texts):
                if text and text.strip():
                    non_empty_texts.append(text)
                else:
                    empty_indices.append(idx)
            
            # Если есть непустые строки, переводим их
            if non_empty_texts:
                # Токенизация с оптимизациями
                inputs = tokenizer(
                    non_empty_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length
                ).to(device)
                
                total_tokens += sum(len(ids) for ids in inputs['input_ids'])
                
                # Генерация с оптимизациями
                with torch.inference_mode():  # Быстрее чем no_grad()
                    outputs = model.generate(
                        **inputs,
                        max_length=max_length,
                        num_beams=num_beams,
                        early_stopping=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True  # Включаем кэширование для ускорения
                    )
                
                # Декодируем результаты
                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            else:
                decoded = []
            
            # Восстанавливаем порядок с пустыми строками
            batch_translations = [""] * len(batch_texts)
            non_empty_idx = 0
            for idx in range(len(batch_texts)):
                if idx not in empty_indices:
                    if non_empty_idx < len(decoded):
                        batch_translations[idx] = decoded[non_empty_idx]
                    non_empty_idx += 1
            
            translations.extend(batch_translations)
            
            # Очищаем кэш CUDA если нужно
            if device == "cuda":
                torch.cuda.empty_cache()
    
    except Exception as e:
        logger.error(f"Ошибка при переводе чанка: {str(e)}")
        # Возвращаем пустые строки для оставшихся текстов
        translations.extend([""] * (len(texts) - len(translations)))
    
    # Замеряем время
    if end_time:
        end_time.record()
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time) / 1000  # конвертируем в секунды
    else:
        elapsed_time = 1.0  # заглушка для CPU
    
    # Вычисляем скорость
    speed = len(texts) / elapsed_time if elapsed_time > 0 else 0.0
    
    logger.info(
        f"Перевод чанка завершен: {len(translations)} предложений, "
        f"{total_tokens} токенов, "
        f"скорость: {speed:.2f} предл./сек"
    )
    
    return translations, round(speed, 2)

def translate_texts(
    texts: List[str],
    model: torch.nn.Module,
    tokenizer,
    batch_size: int = 8,
    max_length: int = 512,
    num_beams: int = 4,
    device: Optional[str] = None,
    show_progress: bool = False,
    n_jobs: int = -1
) -> Tuple[List[str], float]:
    """
    Параллельно переводит список текстов, разбивая их на чанки
    
    Args:
        texts: Список текстов для перевода
        model: Модель для перевода
        tokenizer: Токенизатор
        batch_size: Размер батча
        max_length: Максимальная длина последовательности
        num_beams: Количество лучей для beam search
        device: Устройство для вычислений ('cuda' или 'cpu')
        show_progress: Показывать ли прогресс-бар
        n_jobs: Количество процессов (-1 для использования всех ядер)
        
    Returns:
        Tuple[List[str], float]: (переводы, средняя скорость перевода в предл./сек)
    """
    if not texts:
        return [], 0.0
    
    # Определяем количество процессов
    if n_jobs == -1:
        n_jobs = torch.cuda.device_count() if torch.cuda.is_available() else 1
    n_jobs = max(1, min(n_jobs, len(texts)))  # Не больше чем текстов
    
    # Отключаем многопроцессорную обработку для моделей Marian
    # из-за проблем с сериализацией токенизатора
    if hasattr(tokenizer, 'spm_files'):
        logger.info("Обнаружена модель Marian, отключаем многопроцессорную обработку")
        n_jobs = 1
    
    # Если один процесс или мало текстов, используем обычный перевод
    if n_jobs == 1 or len(texts) < 1000:
        return translate_chunk(
            texts=texts,
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            num_beams=num_beams,
            device=device,
            show_progress=show_progress
        )
    
    # Разбиваем тексты на чанки для параллельной обработки
    chunk_size = math.ceil(len(texts) / n_jobs)
    chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
    
    logger.info(f"Запуск параллельного перевода с {n_jobs} процессами")
    
    # Создаем частичную функцию с фиксированными параметрами
    translate_func = partial(
        translate_chunk,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        num_beams=num_beams,
        device=device,
        show_progress=False  # Отключаем прогресс-бар для параллельных процессов
    )
    
    # Запускаем параллельную обработку
    all_translations = []
    total_speed = 0
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(translate_func, chunk) for chunk in chunks]
        
        # Собираем результаты с прогресс-баром
        if show_progress:
            futures = tqdm(futures, desc="Обработка чанков", total=len(chunks))
        
        for future in futures:
            try:
                translations, speed = future.result()
                all_translations.extend(translations)
                total_speed += speed
            except Exception as e:
                logger.error(f"Ошибка при параллельном переводе: {str(e)}")
                # Добавляем пустые строки для неудачного чанка
                all_translations.extend([""] * chunk_size)
    
    # Проверяем, что количество переводов совпадает
    if len(all_translations) != len(texts):
        logger.warning(
            f"Несоответствие количества переводов: получено {len(all_translations)}, "
            f"ожидалось {len(texts)}"
        )
        # Обрезаем лишние или добавляем недостающие
        if len(all_translations) > len(texts):
            all_translations = all_translations[:len(texts)]
        else:
            all_translations.extend([""] * (len(texts) - len(all_translations)))
    
    # Вычисляем среднюю скорость
    avg_speed = total_speed / n_jobs if n_jobs > 0 else 0.0
    
    logger.info(
        f"Параллельный перевод завершен: {len(all_translations)} предложений, "
        f"средняя скорость: {avg_speed:.2f} предл./сек"
    )
    
    return all_translations, round(avg_speed, 2)

def translate_table(table: List[List[str]], model, tokenizer, **kwargs) -> List[List[str]]:
    """
    Переводит таблицу (список списков строк)
    
    Args:
        table: Таблица для перевода
        model: Модель для перевода
        tokenizer: Токенизатор
        **kwargs: Дополнительные аргументы для translate_texts
        
    Returns:
        List[List[str]]: Переведенная таблица
    """
    # Собираем все непустые строки для перевода
    texts_to_translate = []
    cell_mapping = {}  # индекс в плоском списке -> (row, col)
    
    for i, row in enumerate(table):
        for j, cell in enumerate(row):
            if cell and cell.strip():
                cell_mapping[len(texts_to_translate)] = (i, j)
                texts_to_translate.append(cell)
    
    # Переводим все строки разом
    if texts_to_translate:
        translations, _ = translate_texts(texts_to_translate, model, tokenizer, **kwargs)
        
        # Создаем копию таблицы для результата
        result = [[cell for cell in row] for row in table]
        
        # Заполняем переводы
        for idx, translation in enumerate(translations):
            if idx in cell_mapping:
                row, col = cell_mapping[idx]
                result[row][col] = translation
        
        return result
    
    return table  # Возвращаем исходную таблицу если нечего переводить
