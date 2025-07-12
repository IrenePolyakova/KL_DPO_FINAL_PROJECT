@echo off
echo ===================================
echo  Быстрая проверка Conda
echo ===================================

echo Проверка наличия conda в PATH...
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Conda не найдена в PATH
    echo.
    echo Возможные решения:
    echo 1. Установите Miniconda: https://docs.conda.io/en/latest/miniconda.html
    echo 2. Добавьте conda в PATH вручную
    echo 3. Используйте Anaconda Prompt вместо обычной командной строки
    echo.
    echo Проверка альтернативных путей...
    if exist "C:\ProgramData\miniconda3\Scripts\conda.exe" (
        echo [INFO] Найден Miniconda в: C:\ProgramData\miniconda3\Scripts\
        echo Добавьте этот путь в PATH
    )
    if exist "C:\Users\%USERNAME%\miniconda3\Scripts\conda.exe" (
        echo [INFO] Найден Miniconda в: C:\Users\%USERNAME%\miniconda3\Scripts\
        echo Добавьте этот путь в PATH
    )
    if exist "C:\ProgramData\Anaconda3\Scripts\conda.exe" (
        echo [INFO] Найден Anaconda в: C:\ProgramData\Anaconda3\Scripts\
        echo Добавьте этот путь в PATH
    )
    if exist "C:\Users\%USERNAME%\Anaconda3\Scripts\conda.exe" (
        echo [INFO] Найден Anaconda в: C:\Users\%USERNAME%\Anaconda3\Scripts\
        echo Добавьте этот путь в PATH
    )
    pause
    exit /b 1
)

echo [OK] Conda найдена в PATH
echo.

echo Получение версии conda...
conda --version 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] Не удалось получить версию conda
) else (
    echo [OK] Conda работает
)

echo.
echo Проверка окружений...
conda env list 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] Не удалось получить список окружений
) else (
    echo [OK] Список окружений получен
)

echo.
echo Проверка окружения translation_env...
conda env list | findstr /C:"translation_env" >nul
if %errorlevel% neq 0 (
    echo [INFO] Окружение translation_env не найдено
    echo Используйте setup_conda_env.bat для создания
) else (
    echo [OK] Окружение translation_env найдено
)

echo.
echo ===================================
echo Проверка завершена
echo ===================================
pause
