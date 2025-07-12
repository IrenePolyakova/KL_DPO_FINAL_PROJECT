import os
import zipfile
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def load_model_from_zip(zip_file, extract_path):
    """Загружает модель из ZIP архива"""
    os.makedirs(extract_path, exist_ok=True)
    
    # Сохраняем ZIP во временный файл
    with open(os.path.join(extract_path, "temp.zip"), "wb") as f:
        f.write(zip_file.read())
    
    # Разархивируем
    with zipfile.ZipFile(os.path.join(extract_path, "temp.zip"), 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    # Ищем checkpoint
    model_path = None
    for root, dirs, files in os.walk(extract_path):
        if "pytorch_model.bin" in files:
            model_path = root
            break
    
    if model_path is None:
        raise FileNotFoundError("Не найден файл модели (pytorch_model.bin) в архиве")
    
    # Загружаем модель и токенизатор
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return model, tokenizer