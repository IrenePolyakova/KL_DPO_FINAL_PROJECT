@echo off
echo ===================================
echo  File Preparation Dependencies Setup
echo ===================================

REM Проверка виртуального окружения
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run start.bat and choose option 1 to create environment first.
    pause
    exit /b 1
)

REM Активация виртуального окружения
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing/updating dependencies for File Preparation App...
echo.

REM Основные зависимости
echo Installing core dependencies...
pip install streamlit>=1.24.0 --upgrade
pip install pandas>=1.5.3 --upgrade

REM Зависимости для работы с Excel файлами
echo Installing Excel support dependencies...
pip install openpyxl>=3.0.0 --upgrade
pip install xlrd>=2.0.0

REM Дополнительные зависимости для работы с различными форматами
echo Installing additional format support...
pip install xlsxwriter>=3.0.0
pip install chardet>=5.0.0

REM Проверка установки всех библиотек
echo.
echo ===================================
echo Verifying installations...
echo ===================================

python -c "
import sys
import importlib

def check_package(package_name, display_name=None):
    if display_name is None:
        display_name = package_name
    
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'Unknown')
        print(f'✓ {display_name} {version} - OK')
        return True
    except ImportError:
        print(f'✗ {display_name} - NOT FOUND')
        return False

# Проверяем все необходимые пакеты
packages = [
    ('streamlit', 'Streamlit'),
    ('pandas', 'Pandas'),
    ('openpyxl', 'openpyxl'),
    ('xlrd', 'xlrd'),
    ('xlsxwriter', 'xlsxwriter'),
    ('chardet', 'chardet'),
    ('pathlib', 'pathlib'),
    ('io', 'io (built-in)'),
    ('zipfile', 'zipfile (built-in)'),
    ('os', 'os (built-in)'),
    ('logging', 'logging (built-in)')
]

print('Checking File Preparation App dependencies:')
print('-' * 50)

all_ok = True
for package, display in packages:
    if not check_package(package, display):
        all_ok = False

print('-' * 50)
if all_ok:
    print('✅ All dependencies are installed and ready!')
else:
    print('❌ Some dependencies are missing. Please install them manually.')
    sys.exit(1)
"

if %errorlevel% neq 0 (
    echo ERROR: Some dependencies are missing!
    echo Please check the output above and install missing packages manually.
    pause
    exit /b 1
)

echo.
echo ===================================
echo Setup completed successfully!
echo ===================================
echo.
echo File Preparation App is ready to use.
echo.
echo To run the app:
echo   run_file_preparation.bat
echo.
echo Or from the main menu:
echo   start.bat → choose File Preparation option
echo.
pause
