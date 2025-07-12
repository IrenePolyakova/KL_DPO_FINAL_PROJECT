@echo off
echo ===================================
echo  Translation Backup & Resume System
echo ===================================

REM Проверка виртуального окружения
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run start.bat and choose option 1 first.
    pause
    exit /b 1
)

REM Активация виртуального окружения
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Создание директорий для резервных копий
echo Creating backup directories...
if not exist "translation_backups" mkdir translation_backups
if not exist "translation_backups\models" mkdir translation_backups\models
if not exist "translation_backups\results" mkdir translation_backups\results
if not exist "translation_backups\progress" mkdir translation_backups\progress

REM Настройка переменных окружения для сохранения результатов
echo Setting up backup environment variables...
set TRANSLATION_BACKUP_ENABLED=1
set TRANSLATION_BACKUP_DIR=translation_backups
set TRANSLATION_AUTO_SAVE_INTERVAL=10
set TRANSLATION_RESUME_ON_ERROR=1

REM Создание конфигурационного файла для резервного копирования
echo Creating backup configuration...
echo [backup] > translation_backup_config.ini
echo enabled = true >> translation_backup_config.ini
echo auto_save_interval = 10 >> translation_backup_config.ini
echo resume_on_error = true >> translation_backup_config.ini
echo backup_dir = translation_backups >> translation_backup_config.ini
echo save_model_results = true >> translation_backup_config.ini
echo save_progress = true >> translation_backup_config.ini

REM Создание скрипта для очистки старых резервных копий
echo Creating cleanup script for old backups...
echo import os > cleanup_old_backups.py
echo import time >> cleanup_old_backups.py
echo from pathlib import Path >> cleanup_old_backups.py
echo. >> cleanup_old_backups.py
echo def cleanup_old_backups(backup_dir="translation_backups", days_to_keep=7): >> cleanup_old_backups.py
echo     """Удаляет резервные копии старше указанного количества дней""" >> cleanup_old_backups.py
echo     if not os.path.exists(backup_dir): >> cleanup_old_backups.py
echo         return >> cleanup_old_backups.py
echo. >> cleanup_old_backups.py
echo     current_time = time.time() >> cleanup_old_backups.py
echo     cutoff_time = current_time - (days_to_keep * 24 * 60 * 60) >> cleanup_old_backups.py
echo. >> cleanup_old_backups.py
echo     for root, dirs, files in os.walk(backup_dir): >> cleanup_old_backups.py
echo         for file in files: >> cleanup_old_backups.py
echo             file_path = os.path.join(root, file) >> cleanup_old_backups.py
echo             if os.path.getmtime(file_path) ^< cutoff_time: >> cleanup_old_backups.py
echo                 try: >> cleanup_old_backups.py
echo                     os.remove(file_path) >> cleanup_old_backups.py
echo                     print(f"Deleted old backup: {file_path}") >> cleanup_old_backups.py
echo                 except: >> cleanup_old_backups.py
echo                     pass >> cleanup_old_backups.py
echo. >> cleanup_old_backups.py
echo if __name__ == "__main__": >> cleanup_old_backups.py
echo     cleanup_old_backups() >> cleanup_old_backups.py

echo.
echo ===================================
echo Backup System Setup Completed!
echo ===================================
echo.
echo Features enabled:
echo - Automatic saving every 10 translations
echo - Model-specific result preservation
echo - Progress tracking and resume capability
echo - Error recovery with partial results
echo - Automatic cleanup of old backups
echo.
echo Backup directory: translation_backups\
echo Configuration: translation_backup_config.ini
echo.
echo The system will automatically:
echo 1. Save results from each model separately
echo 2. Resume from last checkpoint on error
echo 3. Preserve completed translations
echo 4. Clean up old backups automatically
echo.
pause
