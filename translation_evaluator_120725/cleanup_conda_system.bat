@echo off
chcp 65001 > nul
echo ===================================
echo  Очистка системы (Conda окружение)
echo ===================================

REM Очистка кэша Python
echo Очистка кэша Python...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
for /r . %%f in (*.pyc) do @if exist "%%f" del /q "%%f"
for /r . %%f in (*.pyo) do @if exist "%%f" del /q "%%f"

REM Очистка кэша pip
echo Очистка кэша pip...
if exist "%USERPROFILE%\AppData\Local\pip\Cache" (
    rd /s /q "%USERPROFILE%\AppData\Local\pip\Cache" 2>nul
)

REM Очистка кэша conda
echo Очистка кэша conda...
conda clean --all -y 2>nul

REM Очистка кэша PyTorch
echo Очистка кэша PyTorch...
if exist "%USERPROFILE%\.cache\torch" (
    rd /s /q "%USERPROFILE%\.cache\torch" 2>nul
)
if exist "torch_cache" (
    rd /s /q "torch_cache" 2>nul
    mkdir torch_cache
)

REM Очистка кэша HuggingFace
echo Очистка кэша HuggingFace...
if exist "%USERPROFILE%\.cache\huggingface" (
    rd /s /q "%USERPROFILE%\.cache\huggingface" 2>nul
)
if exist "transformers_cache" (
    rd /s /q "transformers_cache" 2>nul
    mkdir transformers_cache
)
if exist "huggingface_cache" (
    rd /s /q "huggingface_cache" 2>nul
    mkdir huggingface_cache
)

REM Очистка временных файлов Streamlit
echo Очистка временных файлов Streamlit...
if exist ".streamlit" (
    if exist ".streamlit\logs" (
        rd /s /q ".streamlit\logs" 2>nul
    )
)

REM Очистка временных файлов Windows
echo Очистка временных файлов Windows...
if exist "%TEMP%\streamlit*" (
    del /q "%TEMP%\streamlit*" 2>nul
)

REM Очистка CUDA кэша
echo Очистка CUDA кэша...
if exist "%USERPROFILE%\.nv" (
    rd /s /q "%USERPROFILE%\.nv" 2>nul
)

echo.
echo ===================================
echo Информация о системе
echo ===================================
echo.

REM Дисковое пространство
echo Свободное место на диске:
for /f "tokens=3" %%i in ('dir /-c ^| findstr /i "bytes free"') do echo Свободно: %%i байт
echo.

REM Информация о памяти
echo Информация о памяти:
wmic OS get TotalVisibleMemorySize,FreePhysicalMemory /value | findstr "=" 2>nul || (
    echo Общая память: %computername% 
    systeminfo | findstr /i "Total Physical Memory" 2>nul || echo Информация недоступна
    systeminfo | findstr /i "Available Physical Memory" 2>nul || echo Информация недоступна
)
echo.

REM Информация о процессоре
echo Информация о процессоре:
echo %PROCESSOR_IDENTIFIER%
echo Ядер: %NUMBER_OF_PROCESSORS%
echo.

REM Информация о GPU
echo Информация о GPU:
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits 2>nul || (
    echo NVIDIA GPU не найден или драйверы не установлены
    dxdiag /t dxdiag_temp.txt 2>nul
    if exist dxdiag_temp.txt (
        findstr /i "Card name" dxdiag_temp.txt | findstr /v "Microsoft"
        del dxdiag_temp.txt 2>nul
    )
)
echo.

REM Проверка conda окружения
echo Статус conda окружения:
conda --version 2>nul && (
    echo [OK] Conda установлена
    echo Активные окружения:
    conda env list | findstr "*"
    echo.
    echo Окружение translation_env:
    conda env list | findstr "translation_env" >nul && echo [OK] Найдено || echo [ERROR] Не найдено
) || echo [ERROR] Conda не найдена

REM Проверка Python и библиотек
echo.
echo Проверка Python и библиотек:
conda activate translation_env 2>nul && (
    python --version 2>nul && echo [OK] Python работает || echo [ERROR] Python не работает
    python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>nul && echo [OK] PyTorch || echo [ERROR] PyTorch
    python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')" 2>nul
    python -c "import transformers; print(f'Transformers: {transformers.__version__}')" 2>nul && echo [OK] Transformers || echo [ERROR] Transformers
    python -c "import streamlit; print(f'Streamlit: {streamlit.__version__}')" 2>nul && echo [OK] Streamlit || echo [ERROR] Streamlit
    conda deactivate 2>nul
) || echo [ERROR] Не удалось активировать окружение translation_env

REM Информация о файлах проекта
echo.
echo Структура проекта:
if exist "uploaded_finetuned_models" (
    echo [Модели] Найдены файлы моделей:
    dir "uploaded_finetuned_models" /b | findstr /i ".zip" | findstr /n "^"
) else (
    echo [Модели] Папка моделей не найдена
)

if exist "data" (
    echo [Данные] Найдены файлы данных:
    dir "data" /b | findstr /i ".csv .docx .xlsx" | findstr /n "^"
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
echo - Используйте conda окружение для лучшей изоляции
echo.
pause
