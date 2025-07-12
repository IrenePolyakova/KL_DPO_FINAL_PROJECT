@echo off
chcp 65001 > nul
echo ===================================
echo  Очистка и оптимизация системы
echo ===================================

REM Очистка кэша Python
echo Очистка кэша Python...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
for /r . %%f in (*.pyc) do @if exist "%%f" del /q "%%f"
for /r . %%f in (*.pyo) do @if exist "%%f" del /q "%%f"

REM Очистка кэша моделей и загрузок
echo Очистка кэша PyTorch...
if exist "%USERPROFILE%\.cache\torch" (
    rd /s /q "%USERPROFILE%\.cache\torch" 2>nul
)
if exist "torch_cache" (
    rd /s /q "torch_cache" 2>nul
    mkdir torch_cache
)

echo Очистка кэша Transformers...
if exist "%USERPROFILE%\.cache\huggingface" (
    rd /s /q "%USERPROFILE%\.cache\huggingface\transformers" 2>nul
)
if exist "transformers_cache" (
    rd /s /q "transformers_cache" 2>nul
    mkdir transformers_cache
)

echo Очистка кэша HuggingFace...
if exist "%USERPROFILE%\.cache\huggingface\hub" (
    rd /s /q "%USERPROFILE%\.cache\huggingface\hub" 2>nul
)
if exist "huggingface_cache" (
    rd /s /q "huggingface_cache" 2>nul
    mkdir huggingface_cache
)

REM Очистка временных файлов Streamlit
if exist ".streamlit" (
    if exist ".streamlit\logs" (
        echo Очистка логов Streamlit...
        rd /s /q ".streamlit\logs"
    )
)

REM Очистка временных файлов Windows
if exist "%TEMP%\streamlit*" (
    echo Очистка временных файлов Windows...
    del /q "%TEMP%\streamlit*" 2>nul
)

echo.
echo ===================================
echo Информация о системе
echo ===================================
echo.

echo Свободное место на диске:
for /f "tokens=3" %%i in ('dir /-c ^| findstr /i "bytes free"') do echo Свободно: %%i байт
echo.

echo Информация о памяти:
powershell -command "Get-CimInstance -ClassName Win32_OperatingSystem | Select-Object @{Name='Общая память (GB)';Expression={[math]::round($_.TotalVisibleMemorySize/1MB,2)}}, @{Name='Свободная память (GB)';Expression={[math]::round($_.FreePhysicalMemory/1MB,2)}} | Format-List" 2>nul || echo Информация о памяти недоступна
echo.

echo Информация о процессоре:
echo %PROCESSOR_IDENTIFIER%
echo Ядер: %NUMBER_OF_PROCESSORS%
echo.

echo Информация о GPU:
where nvidia-smi >nul 2>&1 && (
    echo NVIDIA GPU найден:
    nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>nul || echo Не удалось получить информацию
) || echo NVIDIA GPU не найден
echo.

echo Статус Python окружения:
if exist "venv\Scripts\activate.bat" (
    echo [OK] Виртуальное окружение venv найдено
    call venv\Scripts\activate.bat && python --version 2>nul && echo [OK] Python работает || echo [ERROR] Python не работает
) else (
    echo [INFO] Виртуальное окружение venv не найдено
)

REM Проверка conda окружения
where conda >nul 2>&1 && (
    echo [INFO] Conda найдена
    conda env list | findstr /C:"translation_env" >nul && echo [OK] Conda окружение translation_env найдено || echo [INFO] Conda окружение translation_env не найдено
) || echo [INFO] Conda не найдена
echo.

echo Структура проекта:
if exist "uploaded_finetuned_models" (
    echo [Модели] Найдены файлы моделей:
    dir "uploaded_finetuned_models" | findstr /i ".zip"
) else (
    echo [Модели] Папка моделей не найдена
)
if exist "data" (
    echo [Данные] Найдены файлы данных:
    dir "data" | findstr /i ".csv .docx .xlsx"
) else (
    echo [Данные] Папка данных не найдена
)
echo.

echo ===================================
echo Очистка завершена!
echo ===================================
echo.
echo Рекомендации для улучшения производительности:
echo - Закройте ненужные приложения
echo - Обеспечьте минимум 10GB свободного места
echo - Используйте SSD накопитель если возможно
echo - Увеличьте виртуальную память для больших моделей
echo - Перезапустите приложение после очистки
echo - Рекомендуется использовать conda окружение
echo.
pause
