#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль для сохранения и восстановления результатов перевода
Обеспечивает сохранение промежуточных результатов при работе с несколькими моделями
"""

import os
import json
import pickle
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging
import hashlib

logger = logging.getLogger(__name__)

class TranslationBackupManager:
    """Менеджер резервного копирования переводов"""
    
    def __init__(self, backup_dir="translation_backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        
        # Создаем подпапки
        (self.backup_dir / "models").mkdir(exist_ok=True)
        (self.backup_dir / "results").mkdir(exist_ok=True)
        (self.backup_dir / "progress").mkdir(exist_ok=True)
        (self.backup_dir / "sessions").mkdir(exist_ok=True)
        
        self.current_session_id = None
        self.auto_save_interval = 10  # Сохранять каждые 10 переводов
        
    def create_session(self, source_texts, model_names=None):
        """
        Создает новую сессию перевода
        
        Args:
            source_texts: Список исходных текстов
            model_names: Список имен моделей для перевода
        
        Returns:
            str: ID сессии
        """
        # Создаем уникальный ID сессии на основе содержимого
        content_hash = hashlib.md5(str(source_texts).encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"session_{timestamp}_{content_hash}"
        
        self.current_session_id = session_id
        
        # Сохраняем информацию о сессии
        session_info = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "source_texts": source_texts,
            "model_names": model_names or [],
            "total_texts": len(source_texts),
            "completed_models": [],
            "failed_models": [],
            "progress": {}
        }
        
        session_file = self.backup_dir / "sessions" / f"{session_id}.json"
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Created translation session: {session_id}")
        return session_id
    
    def save_model_results(self, session_id, model_name, translations, metadata=None):
        """
        Сохраняет результаты перевода для конкретной модели
        
        Args:
            session_id: ID сессии
            model_name: Название модели
            translations: Список переводов
            metadata: Дополнительные метаданные
        """
        if not session_id:
            session_id = self.current_session_id
            
        if not session_id:
            logger.warning("No session ID provided for saving results")
            return
        
        # Создаем безопасное имя файла для модели
        safe_model_name = "".join(c for c in model_name if c.isalnum() or c in ('-', '_')).rstrip()
        
        # Сохраняем результаты модели
        model_results = {
            "model_name": model_name,
            "session_id": session_id,
            "saved_at": datetime.now().isoformat(),
            "translations": translations,
            "total_translations": len(translations),
            "metadata": metadata or {}
        }
        
        # Сохраняем в JSON формате
        results_file = self.backup_dir / "models" / f"{session_id}_{safe_model_name}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(model_results, f, ensure_ascii=False, indent=2)
        
        # Также сохраняем в формате pickle для сложных объектов
        pickle_file = self.backup_dir / "models" / f"{session_id}_{safe_model_name}.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(model_results, f)
        
        # Обновляем информацию о сессии
        self._update_session_progress(session_id, model_name, len(translations), "completed")
        
        logger.info(f"Saved results for model {model_name} in session {session_id}")
    
    def save_progress(self, session_id, model_name, completed_count, total_count, 
                     partial_translations=None):
        """
        Сохраняет прогресс перевода для возможности восстановления
        
        Args:
            session_id: ID сессии
            model_name: Название модели
            completed_count: Количество завершенных переводов
            total_count: Общее количество текстов для перевода
            partial_translations: Частичные результаты перевода
        """
        if not session_id:
            session_id = self.current_session_id
            
        progress_info = {
            "session_id": session_id,
            "model_name": model_name,
            "completed_count": completed_count,
            "total_count": total_count,
            "progress_percentage": (completed_count / total_count) * 100 if total_count > 0 else 0,
            "saved_at": datetime.now().isoformat(),
            "partial_translations": partial_translations or []
        }
        
        safe_model_name = "".join(c for c in model_name if c.isalnum() or c in ('-', '_')).rstrip()
        progress_file = self.backup_dir / "progress" / f"{session_id}_{safe_model_name}_progress.json"
        
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved progress for {model_name}: {completed_count}/{total_count}")
    
    def load_model_results(self, session_id, model_name):
        """
        Загружает сохраненные результаты модели
        
        Args:
            session_id: ID сессии
            model_name: Название модели
            
        Returns:
            dict: Результаты перевода или None если не найдены
        """
        safe_model_name = "".join(c for c in model_name if c.isalnum() or c in ('-', '_')).rstrip()
        
        # Пробуем загрузить из pickle файла
        pickle_file = self.backup_dir / "models" / f"{session_id}_{safe_model_name}.pkl"
        if pickle_file.exists():
            try:
                with open(pickle_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load pickle file: {e}")
        
        # Пробуем загрузить из JSON файла
        json_file = self.backup_dir / "models" / f"{session_id}_{safe_model_name}.json"
        if json_file.exists():
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load JSON file: {e}")
        
        return None
    
    def load_progress(self, session_id, model_name):
        """
        Загружает сохраненный прогресс модели
        
        Args:
            session_id: ID сессии
            model_name: Название модели
            
        Returns:
            dict: Информация о прогрессе или None
        """
        safe_model_name = "".join(c for c in model_name if c.isalnum() or c in ('-', '_')).rstrip()
        progress_file = self.backup_dir / "progress" / f"{session_id}_{safe_model_name}_progress.json"
        
        if progress_file.exists():
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load progress file: {e}")
        
        return None
    
    def get_session_info(self, session_id):
        """
        Получает информацию о сессии
        
        Args:
            session_id: ID сессии
            
        Returns:
            dict: Информация о сессии или None
        """
        session_file = self.backup_dir / "sessions" / f"{session_id}.json"
        
        if session_file.exists():
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load session file: {e}")
        
        return None
    
    def list_sessions(self):
        """
        Возвращает список всех сессий
        
        Returns:
            list: Список ID сессий
        """
        sessions_dir = self.backup_dir / "sessions"
        if not sessions_dir.exists():
            return []
        
        sessions = []
        for session_file in sessions_dir.glob("session_*.json"):
            session_id = session_file.stem
            sessions.append(session_id)
        
        return sorted(sessions, reverse=True)  # Новые сессии первыми
    
    def get_completed_models(self, session_id):
        """
        Возвращает список моделей, которые уже завершили перевод в данной сессии
        
        Args:
            session_id: ID сессии
            
        Returns:
            list: Список названий моделей
        """
        models_dir = self.backup_dir / "models"
        completed_models = []
        
        for model_file in models_dir.glob(f"{session_id}_*.json"):
            # Извлекаем название модели из имени файла
            filename = model_file.stem
            model_part = filename.replace(f"{session_id}_", "")
            completed_models.append(model_part)
        
        return completed_models
    
    def cleanup_old_sessions(self, days_to_keep=7):
        """
        Удаляет старые сессии и их файлы
        
        Args:
            days_to_keep: Количество дней для хранения
        """
        import time
        cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
        
        # Очищаем файлы старше cutoff_time
        for subdir in ["sessions", "models", "progress", "results"]:
            dir_path = self.backup_dir / subdir
            if dir_path.exists():
                for file_path in dir_path.iterdir():
                    if file_path.stat().st_mtime < cutoff_time:
                        try:
                            file_path.unlink()
                            logger.info(f"Deleted old backup file: {file_path}")
                        except Exception as e:
                            logger.warning(f"Failed to delete {file_path}: {e}")
    
    def _update_session_progress(self, session_id, model_name, translation_count, status):
        """Обновляет прогресс в файле сессии"""
        session_file = self.backup_dir / "sessions" / f"{session_id}.json"
        
        if session_file.exists():
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    session_info = json.load(f)
                
                # Обновляем прогресс
                session_info["progress"][model_name] = {
                    "status": status,
                    "translation_count": translation_count,
                    "updated_at": datetime.now().isoformat()
                }
                
                if status == "completed":
                    if model_name not in session_info["completed_models"]:
                        session_info["completed_models"].append(model_name)
                elif status == "failed":
                    if model_name not in session_info["failed_models"]:
                        session_info["failed_models"].append(model_name)
                
                # Сохраняем обновленную информацию
                with open(session_file, 'w', encoding='utf-8') as f:
                    json.dump(session_info, f, ensure_ascii=False, indent=2)
                
            except Exception as e:
                logger.warning(f"Failed to update session progress: {e}")

# Глобальный экземпляр менеджера
backup_manager = TranslationBackupManager()

def save_translation_results(session_id, model_name, translations, metadata=None):
    """Сохраняет результаты перевода (удобная функция)"""
    return backup_manager.save_model_results(session_id, model_name, translations, metadata)

def load_translation_results(session_id, model_name):
    """Загружает результаты перевода (удобная функция)"""
    return backup_manager.load_model_results(session_id, model_name)

def create_translation_session(source_texts, model_names=None):
    """Создает сессию перевода (удобная функция)"""
    return backup_manager.create_session(source_texts, model_names)
