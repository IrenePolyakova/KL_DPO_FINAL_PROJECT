#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Интеграция системы резервного копирования с Translation Evaluator
Добавляет возможность сохранения и восстановления результатов перевода
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import logging

try:
    from translation_utils.backup_manager import TranslationBackupManager
except ImportError:
    # Если модуль не найден, создаем заглушку
    class TranslationBackupManager:
        def __init__(self, *args, **kwargs):
            pass
        def create_session(self, *args, **kwargs):
            return None
        def save_model_results(self, *args, **kwargs):
            pass
        def load_model_results(self, *args, **kwargs):
            return None

logger = logging.getLogger(__name__)

def init_backup_system():
    """Инициализирует систему резервного копирования"""
    if 'backup_manager' not in st.session_state:
        st.session_state.backup_manager = TranslationBackupManager()
    
    if 'current_session_id' not in st.session_state:
        st.session_state.current_session_id = None
    
    if 'saved_models' not in st.session_state:
        st.session_state.saved_models = {}

def show_backup_status():
    """Отображает статус резервного копирования в боковой панели"""
    with st.sidebar:
        st.markdown("---")
        st.subheader("🔄 Система резервных копий")
        
        if st.session_state.current_session_id:
            st.success(f"Активная сессия: {st.session_state.current_session_id[:12]}...")
            
            # Показываем сохраненные модели
            if st.session_state.saved_models:
                st.write("Сохраненные модели:")
                for model_name, status in st.session_state.saved_models.items():
                    if status == "completed":
                        st.write(f"✅ {model_name}")
                    elif status == "in_progress":
                        st.write(f"🔄 {model_name}")
                    else:
                        st.write(f"❌ {model_name}")
        else:
            st.info("Сессия не создана")

def create_translation_session(source_texts, model_names=None):
    """
    Создает новую сессию перевода с резервным копированием
    
    Args:
        source_texts: Список исходных текстов
        model_names: Список имен моделей
        
    Returns:
        str: ID сессии
    """
    init_backup_system()
    
    session_id = st.session_state.backup_manager.create_session(
        source_texts, model_names
    )
    
    st.session_state.current_session_id = session_id
    st.session_state.saved_models = {}
    
    logger.info(f"Created backup session: {session_id}")
    
    # Показываем уведомление пользователю
    st.info(f"🔄 Создана сессия резервного копирования: {session_id[:12]}...")
    
    return session_id

def save_model_translation(model_name, translations, metadata=None):
    """
    Сохраняет результаты перевода модели
    
    Args:
        model_name: Название модели
        translations: Список переводов
        metadata: Метаданные
    """
    init_backup_system()
    
    if not st.session_state.current_session_id:
        st.warning("Сессия резервного копирования не создана")
        return
    
    try:
        st.session_state.backup_manager.save_model_results(
            st.session_state.current_session_id,
            model_name,
            translations,
            metadata
        )
        
        # Обновляем статус модели
        st.session_state.saved_models[model_name] = "completed"
        
        st.success(f"✅ Результаты модели {model_name} сохранены")
        logger.info(f"Saved results for model: {model_name}")
        
    except Exception as e:
        st.error(f"❌ Ошибка сохранения результатов модели {model_name}: {str(e)}")
        logger.error(f"Failed to save model results: {e}")

def load_model_translation(model_name, session_id=None):
    """
    Загружает сохраненные результаты перевода модели
    
    Args:
        model_name: Название модели
        session_id: ID сессии (если не указан, используется текущая)
        
    Returns:
        dict: Результаты перевода или None
    """
    init_backup_system()
    
    if not session_id:
        session_id = st.session_state.current_session_id
    
    if not session_id:
        return None
    
    try:
        results = st.session_state.backup_manager.load_model_results(
            session_id, model_name
        )
        
        if results:
            st.info(f"🔄 Загружены сохраненные результаты для модели {model_name}")
            logger.info(f"Loaded saved results for model: {model_name}")
        
        return results
        
    except Exception as e:
        st.error(f"❌ Ошибка загрузки результатов модели {model_name}: {str(e)}")
        logger.error(f"Failed to load model results: {e}")
        return None

def show_session_recovery():
    """Показывает интерфейс восстановления сессий"""
    init_backup_system()
    
    with st.expander("🔄 Восстановление предыдущих сессий"):
        sessions = st.session_state.backup_manager.list_sessions()
        
        if not sessions:
            st.info("Нет сохраненных сессий")
            return
        
        st.write("Доступные сессии:")
        
        for session_id in sessions[:10]:  # Показываем последние 10 сессий
            session_info = st.session_state.backup_manager.get_session_info(session_id)
            
            if session_info:
                created_at = session_info.get('created_at', 'Unknown')
                total_texts = session_info.get('total_texts', 0)
                completed_models = session_info.get('completed_models', [])
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**{session_id[:12]}...** ({created_at[:16]})")
                    st.write(f"Текстов: {total_texts}, Моделей завершено: {len(completed_models)}")
                    if completed_models:
                        st.write(f"Модели: {', '.join(completed_models[:3])}{'...' if len(completed_models) > 3 else ''}")
                
                with col2:
                    if st.button("Восстановить", key=f"restore_{session_id}"):
                        st.session_state.current_session_id = session_id
                        
                        # Загружаем информацию о сохраненных моделях
                        st.session_state.saved_models = {}
                        for model_name in completed_models:
                            st.session_state.saved_models[model_name] = "completed"
                        
                        st.success(f"Сессия {session_id[:12]}... восстановлена")
                        st.experimental_rerun()

def show_backup_management():
    """Показывает интерфейс управления резервными копиями"""
    init_backup_system()
    
    with st.expander("⚙️ Управление резервными копиями"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Текущая сессия")
            if st.session_state.current_session_id:
                st.code(st.session_state.current_session_id)
                
                if st.button("Создать новую сессию"):
                    st.session_state.current_session_id = None
                    st.session_state.saved_models = {}
                    st.success("Готово к созданию новой сессии")
                    st.experimental_rerun()
            else:
                st.info("Сессия не активна")
        
        with col2:
            st.subheader("Очистка")
            days_to_keep = st.number_input(
                "Хранить файлы (дней)", 
                min_value=1, 
                max_value=30, 
                value=7
            )
            
            if st.button("Очистить старые файлы"):
                try:
                    st.session_state.backup_manager.cleanup_old_sessions(days_to_keep)
                    st.success(f"Удалены файлы старше {days_to_keep} дней")
                except Exception as e:
                    st.error(f"Ошибка очистки: {str(e)}")

def get_backup_statistics():
    """Возвращает статистику резервных копий"""
    init_backup_system()
    
    try:
        backup_dir = Path("translation_backups")
        if not backup_dir.exists():
            return {"sessions": 0, "models": 0, "size_mb": 0}
        
        sessions_count = len(list((backup_dir / "sessions").glob("*.json"))) if (backup_dir / "sessions").exists() else 0
        models_count = len(list((backup_dir / "models").glob("*.json"))) if (backup_dir / "models").exists() else 0
        
        # Подсчитываем размер
        total_size = 0
        for file_path in backup_dir.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        size_mb = total_size / (1024 * 1024)
        
        return {
            "sessions": sessions_count,
            "models": models_count,
            "size_mb": round(size_mb, 1)
        }
        
    except Exception as e:
        logger.error(f"Failed to get backup statistics: {e}")
        return {"sessions": 0, "models": 0, "size_mb": 0}

def auto_save_progress(model_name, completed_count, total_count, partial_translations=None):
    """
    Автоматически сохраняет прогресс перевода
    
    Args:
        model_name: Название модели
        completed_count: Количество завершенных переводов
        total_count: Общее количество
        partial_translations: Частичные результаты
    """
    init_backup_system()
    
    if not st.session_state.current_session_id:
        return
    
    try:
        st.session_state.backup_manager.save_progress(
            st.session_state.current_session_id,
            model_name,
            completed_count,
            total_count,
            partial_translations
        )
        
        # Обновляем статус модели
        if model_name not in st.session_state.saved_models:
            st.session_state.saved_models[model_name] = "in_progress"
        
    except Exception as e:
        logger.error(f"Failed to auto-save progress: {e}")

# Decorator для автоматического сохранения результатов
def with_backup(model_name):
    """
    Декоратор для автоматического сохранения результатов перевода
    
    Args:
        model_name: Название модели
    """
    def decorator(translate_func):
        def wrapper(*args, **kwargs):
            try:
                # Выполняем перевод
                result = translate_func(*args, **kwargs)
                
                # Автоматически сохраняем результат
                if result and isinstance(result, (list, tuple)):
                    save_model_translation(model_name, result)
                
                return result
                
            except Exception as e:
                # В случае ошибки, все равно пытаемся сохранить частичные результаты
                logger.error(f"Translation failed for {model_name}: {e}")
                raise
        
        return wrapper
    return decorator
