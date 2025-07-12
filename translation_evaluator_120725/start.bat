@echo off
chcp 65001 > nul
cls

:MENU
echo.
echo ================================================================================
echo                        Translation Evaluator Manager
echo ================================================================================
echo.
echo  === CONDA ОКРУЖЕНИЕ (Рекомендуется) ===
echo  1. [CONDA-SETUP] Создать Conda окружение    - Python 3.9 + GPU поддержка
echo  2. [CONDA-FAST]  Быстрая установка Conda    - CPU версия, быстрая установка
echo  3. [CONDA-RUN]   Запуск (Conda)             - С проверками и диагностикой
echo  4. [CONDA-QUICK] Быстрый запуск (Conda)     - Мгновенный старт без проверок
echo  5. [CONDA-DIAG]  Диагностика (Conda)        - Проверка системы
echo.
echo  === ОБЫЧНЫЕ ОПЦИИ ===
echo  6. [SETUP] Setup Environment        - Создать venv окружение
echo  7. [RUN]   Run Application (Standard) - Запуск с поддержкой 5GB файлов
echo  8. [PERF]  Run Application (Performance) - Запуск 8GB + оптимизации
echo  9. [SMART] Run Application (Smart Port) - Автовыбор порта + 5GB
echo 10. [ULTRA] Run Application (Ultra)   - Максимальная производительность 10GB
echo 11. [BACKUP] Run with Backup System  - Автосохранение + восстановление
echo 12. [FILE]  File Preparation App     - Подготовка файлов для оценки
echo 13. [CLEAN] System Cleanup           - Очистка кэша и оптимизация
echo 14. [STATUS] System Status           - Проверка окружения и системы
echo 15. [FIX]   Fix Dependencies         - Исправление зависимостей
echo 16. [ANALYZE] Analyze Performance    - Анализ скорости перевода
echo 17. [OPTIMIZE] Optimize Performance  - Автооптимизация
echo 18. [UPDATE] Update Dependencies     - Обновление пакетов Python
echo 19. [EXIT]  Exit                     - Закрыть меню
echo.
echo ================================================================================
echo.
set /p choice="Выберите опцию (1-19): "

if "%choice%"=="1" goto CONDA_SETUP
if "%choice%"=="2" goto CONDA_FAST
if "%choice%"=="3" goto CONDA_RUN
if "%choice%"=="4" goto CONDA_QUICK
if "%choice%"=="5" goto CONDA_DIAG
if "%choice%"=="6" goto SETUP_ENV
if "%choice%"=="7" goto RUN_STANDARD
if "%choice%"=="8" goto RUN_PERFORMANCE
if "%choice%"=="9" goto RUN_SMART
if "%choice%"=="10" goto RUN_ULTRA
if "%choice%"=="11" goto RUN_BACKUP
if "%choice%"=="12" goto FILE_PREP
if "%choice%"=="15" goto FIX_DEPS
if "%choice%"=="16" goto ANALYZE_PERF
if "%choice%"=="17" goto OPTIMIZE_PERF
if "%choice%"=="18" goto UPDATE_DEPS
if "%choice%"=="19" goto EXIT
echo Неверный выбор. Попробуйте снова.
pause
goto MENU

:CONDA_SETUP
echo.
echo ===================================
echo Создание Conda окружения...
echo ===================================
call setup_conda_env.bat
pause
goto MENU

:CONDA_FAST
echo.
echo ===================================
echo Быстрая установка Conda окружения...
echo ===================================
call setup_conda_env_fast.bat
pause
goto MENU

:CONDA_RUN
echo.
echo ===================================
echo Запуск приложения (Conda)...
echo ===================================
call run_app_conda.bat
pause
goto MENU

:CONDA_QUICK
echo.
echo ===================================
echo Быстрый запуск приложения (Conda)...
echo ===================================
call run_app_conda_fast.bat
pause
goto MENU

:CONDA_DIAG
echo.
echo ===================================
echo Диагностика системы (Conda)...
echo ===================================
call diagnose_conda.bat
pause
goto MENU

:SETUP_ENV
echo.
echo ===================================
echo Setting up environment...
echo ===================================
call setup_env.bat
pause
goto MENU

:RUN_STANDARD
echo.
echo ===================================
echo Launching standard application...
echo ===================================
call run_app.bat
pause
goto MENU

:RUN_PERFORMANCE
echo.
echo ===================================
echo Launching high-performance mode...
echo ===================================
call run_app_performance.bat
pause
goto MENU

:RUN_SMART
echo.
echo ===================================
echo Launching smart port selection...
echo ===================================
call run_app_smart.bat
pause
goto MENU

:RUN_ULTRA
echo.
echo ===================================
echo Launching ultra-performance mode...
echo ===================================
call run_app_ultra.bat
pause
goto MENU

:RUN_BACKUP
echo.
echo ===================================
echo Launching app with backup system...
echo ===================================
call run_app_with_backup.bat
pause
goto MENU

:FILE_PREP
echo.
echo ===================================
echo Launching File Preparation App...
echo ===================================
call run_file_preparation.bat
pause
goto MENU

:CLEANUP
echo.
echo ===================================
echo Running system cleanup...
echo ===================================
call cleanup_system.bat
pause
goto MENU

:STATUS
echo.
echo ===================================
echo System Status Check
echo ===================================
echo.
echo Python version:
python --version 2>nul || echo Python not found!
echo.
echo Virtual environment:
if exist "venv\Scripts\activate.bat" (
    echo [OK] Virtual environment exists
) else (
    echo [ERROR] Virtual environment not found
)
echo.
echo Dependencies status:
call venv\Scripts\activate.bat 2>nul && python -c "import streamlit, torch, transformers; print('[OK] Core dependencies installed')" 2>nul || echo "[ERROR] Dependencies missing"
echo.
echo Disk space:
for /f "tokens=3" %%i in ('dir /-c ^| findstr /i "bytes free"') do echo Free space: %%i bytes
echo.
echo Memory info:
wmic OS get TotalVisibleMemorySize,FreePhysicalMemory /value | findstr "="
echo.
pause
goto MENU

:ANALYZE_PERF
echo.
echo ===================================
echo Analyzing translation performance...
echo ===================================
call analyze_performance.bat
pause
goto MENU

:OPTIMIZE_PERF
echo.
echo ===================================
echo Optimizing translation performance...
echo ===================================
call optimize_performance.bat
pause
goto MENU

:UPDATE_DEPS
echo.
echo ===================================
echo Updating dependencies...
echo ===================================
if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found! Please run setup first.
    pause
    goto MENU
)
call venv\Scripts\activate.bat
echo Updating pip...
python -m pip install --upgrade pip
echo Updating packages...
pip install -r requirements.txt --upgrade
echo Dependencies updated!
pause
goto MENU

:FIX_DEPS
echo.
echo ===================================
echo Fixing dependencies issues...
echo ===================================
call fix_dependencies.bat
pause
goto MENU

:EXIT
echo.
echo Thank you for using Translation Evaluator Manager!
echo.
exit /b 0
