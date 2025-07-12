import os
import zipfile
import torch
import logging
import shutil
from typing import Tuple, Optional
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from functools import lru_cache

logger = logging.getLogger(__name__)

class ModelLoadError(Exception):
    """Ошибка загрузки модели"""
    pass

@lru_cache(maxsize=2)  # Кэшируем последние 2 загруженные модели
def load_model_from_zip(
    zip_file,
    extract_path: str,
    device: Optional[str] = None,
    token: Optional[str] = None  # Заменяем use_auth_token на token
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Загружает модель из ZIP архива
    
    Args:
        zip_file: путь к ZIP файлу или файловый объект
        extract_path: путь для распаковки
        device: устройство для загрузки модели ('cuda' или 'cpu')
        token: токен для загрузки из Hugging Face Hub (заменяет use_auth_token)
        
    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer]: модель и токенизатор
    """
    try:
        extract_path = Path(extract_path)
        extract_path.mkdir(parents=True, exist_ok=True)
        
        # Определяем устройство
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Проверяем, есть ли уже распакованная модель
        model_files = list(extract_path.glob("pytorch_model.bin")) + list(extract_path.glob("model.safetensors"))
        spm_files = list(extract_path.glob("*.spm"))
        
        if not model_files or not spm_files:
            logger.info(f"Распаковка модели в {extract_path}")
            
            # Сохраняем ZIP во временный файл
            zip_path = extract_path / "temp.zip"
            
            # Обрабатываем разные типы входных данных
            if isinstance(zip_file, (str, Path)):
                logger.info(f"Копирование ZIP файла из {zip_file}")
                shutil.copy2(zip_file, zip_path)
            else:
                # Если это файловый объект или UploadedFile, читаем его содержимое
                logger.info("Чтение содержимого ZIP из файлового объекта")
                with open(zip_path, "wb") as f:
                    # Для UploadedFile используем getvalue(), для обычных файлов - read()
                    if hasattr(zip_file, 'getvalue'):
                        f.write(zip_file.getvalue())
                    else:
                        f.write(zip_file.read())
            
            # Проверяем содержимое архива перед распаковкой
            logger.info("Проверка содержимого ZIP архива:")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                logger.info(f"Файлы в архиве: {file_list}")
                
                # Ищем файл модели и .spm файлы в архиве
                model_file = None
                spm_files_in_zip = []
                for file in file_list:
                    if file.endswith('pytorch_model.bin') or file.endswith('model.safetensors'):
                        model_file = file
                    elif file.endswith('.spm'):
                        spm_files_in_zip.append(file)
                
                if not model_file:
                    raise ModelLoadError(
                        f"Файл модели (pytorch_model.bin или model.safetensors) не найден в архиве. "
                        f"Содержимое архива: {file_list}"
                    )
                
                if not spm_files_in_zip:
                    raise ModelLoadError(
                        f"Файлы токенизатора (*.spm) не найдены в архиве. "
                        f"Содержимое архива: {file_list}"
                    )
                
                # Разархивируем
                logger.info("Распаковка архива...")
                zip_ref.extractall(extract_path)
            
            # Удаляем временный ZIP
            zip_path.unlink()
            
            # Ищем распакованные файлы
            logger.info("Поиск распакованных файлов модели...")
            model_files = list(extract_path.rglob("pytorch_model.bin")) + list(extract_path.rglob("model.safetensors"))
            spm_files = list(extract_path.rglob("*.spm"))
            
            if not model_files:
                raise ModelLoadError(
                    f"Файл модели не найден после распаковки в {extract_path}. "
                    f"Содержимое директории: {list(extract_path.rglob('*'))}"
                )
            
            if not spm_files:
                raise ModelLoadError(
                    f"Файлы токенизатора (*.spm) не найдены после распаковки в {extract_path}. "
                    f"Содержимое директории: {list(extract_path.rglob('*'))}"
                )
            
            model_path = model_files[0].parent
            logger.info(f"Найден файл модели в {model_path}")
        
        # Загружаем модель и токенизатор
        logger.info(f"Загрузка модели на {device}")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            extract_path,
            device_map=device,
            token=token  # Используем новый параметр token
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            extract_path,
            token=token  # Используем новый параметр token
        )
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {str(e)}")
        raise ModelLoadError(f"Не удалось загрузить модель: {str(e)}")
        
def cleanup_model_cache():
    """Очищает кэш загруженных моделей"""
    load_model_from_zip.cache_clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()