# Скрипт для отката к последней рабочей версии

param(
    [Parameter(Mandatory=$false)]
    [string]$CommitHash,
    
    [Parameter(Mandatory=$false)]
    [switch]$ShowHistory,
    
    [Parameter(Mandatory=$false)]
    [switch]$Soft
)

# Показываем историю коммитов
if ($ShowHistory) {
    Write-Host "История последних 10 коммитов:"
    git log --oneline -n 10
    exit 0
}

# Если хеш коммита не указан, откатываемся на один коммит назад
if (-not $CommitHash) {
    $CommitHash = "HEAD~1"
}

# Проверяем существование коммита
$commitExists = git rev-parse --verify $CommitHash 2>$null
if (-not $commitExists) {
    Write-Host "Ошибка: Указанный коммит не найден"
    exit 1
}

# Выполняем откат
if ($Soft) {
    # Мягкий откат - сохраняет изменения как неподтвержденные
    git reset --soft $CommitHash
    Write-Host "Выполнен мягкий откат к коммиту $CommitHash"
    Write-Host "Изменения сохранены как неподтвержденные"
} else {
    # Жесткий откат - полностью удаляет изменения
    git reset --hard $CommitHash
    Write-Host "Выполнен жесткий откат к коммиту $CommitHash"
    Write-Host "Все изменения после этого коммита удалены"
}

# Синхронизируем с GitHub
Write-Host "Синхронизация с GitHub..."
git push -f origin main

Write-Host "Откат завершен успешно!" 