# Скрипт для автоматической синхронизации с GitHub

# Проверяем наличие изменений
$changes = git status --porcelain

if ($changes) {
    Write-Host "Обнаружены изменения. Начинаем синхронизацию..."
    
    # Добавляем все изменения
    git add .
    
    # Получаем текущую дату и время для коммита
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    
    # Создаем коммит
    git commit -m "Auto-sync update: $timestamp"
    
    # Отправляем изменения на GitHub
    git push origin main
    
    Write-Host "Синхронизация завершена успешно!"
} else {
    Write-Host "Изменений не обнаружено."
} 