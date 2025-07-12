#!/bin/bash

# Скрипт для автоматической синхронизации с GitHub

# Проверяем наличие изменений
changes=$(git status --porcelain)

if [ -n "$changes" ]; then
    echo "Обнаружены изменения. Начинаем синхронизацию..."
    
    # Добавляем все изменения
    git add .
    
    # Получаем текущую дату и время для коммита
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Создаем коммит
    git commit -m "Auto-sync update: $timestamp"
    
    # Отправляем изменения на GitHub
    git push origin main
    
    echo "Синхронизация завершена успешно!"
else
    echo "Изменений не обнаружено."
fi 