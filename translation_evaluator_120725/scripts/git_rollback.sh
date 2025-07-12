#!/bin/bash

# Скрипт для отката к последней рабочей версии

# Обработка параметров
COMMIT_HASH=""
SHOW_HISTORY=false
SOFT_RESET=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--history)
            SHOW_HISTORY=true
            shift
            ;;
        -s|--soft)
            SOFT_RESET=true
            shift
            ;;
        *)
            COMMIT_HASH="$1"
            shift
            ;;
    esac
done

# Показываем историю коммитов
if [ "$SHOW_HISTORY" = true ]; then
    echo "История последних 10 коммитов:"
    git log --oneline -n 10
    exit 0
fi

# Если хеш коммита не указан, откатываемся на один коммит назад
if [ -z "$COMMIT_HASH" ]; then
    COMMIT_HASH="HEAD~1"
fi

# Проверяем существование коммита
if ! git rev-parse --verify "$COMMIT_HASH" >/dev/null 2>&1; then
    echo "Ошибка: Указанный коммит не найден"
    exit 1
fi

# Выполняем откат
if [ "$SOFT_RESET" = true ]; then
    # Мягкий откат - сохраняет изменения как неподтвержденные
    git reset --soft "$COMMIT_HASH"
    echo "Выполнен мягкий откат к коммиту $COMMIT_HASH"
    echo "Изменения сохранены как неподтвержденные"
else
    # Жесткий откат - полностью удаляет изменения
    git reset --hard "$COMMIT_HASH"
    echo "Выполнен жесткий откат к коммиту $COMMIT_HASH"
    echo "Все изменения после этого коммита удалены"
fi

# Синхронизируем с GitHub
echo "Синхронизация с GitHub..."
git push -f origin main

echo "Откат завершен успешно!" 