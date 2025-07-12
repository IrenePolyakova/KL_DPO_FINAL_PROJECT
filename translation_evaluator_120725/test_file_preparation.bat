@echo off
echo ===================================
echo  File Preparation App Test
echo ===================================

REM Проверка виртуального окружения
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run: start.bat → option 1
    pause
    exit /b 1
)

REM Активация виртуального окружения
call venv\Scripts\activate.bat

REM Проверка файла приложения
if not exist "file_preparation_app.py" (
    echo ERROR: file_preparation_app.py not found!
    echo Please make sure the file exists in the current directory.
    pause
    exit /b 1
)

echo Testing File Preparation App dependencies and functionality...
echo.

REM Тест импорта всех модулей
python -c "
print('Testing imports...')
try:
    import streamlit as st
    import pandas as pd
    import io
    import os
    import zipfile
    from pathlib import Path
    import logging
    
    # Тест функций из файла
    import sys
    sys.path.insert(0, '.')
    
    # Импорт функций из file_preparation_app
    from file_preparation_app import (
        convert_xlsx_to_csv_data,
        detect_columns,
        clean_text,
        prepare_files_data,
        create_instructions
    )
    
    print('✓ All imports successful!')
    
    # Тест базовой функциональности
    print('Testing basic functionality...')
    
    # Тест функции очистки текста
    test_text = clean_text('  Hello World  ')
    assert test_text == 'Hello World', 'Text cleaning failed'
    print('✓ Text cleaning works')
    
    # Тест создания тестовых данных
    test_df = pd.DataFrame({
        'source': ['Hello', 'World'],
        'target': ['Привет', 'Мир']
    })
    
    src_col, tgt_col = detect_columns(test_df)
    assert src_col == 'source', 'Source column detection failed'
    assert tgt_col == 'target', 'Target column detection failed'
    print('✓ Column detection works')
    
    # Тест подготовки данных
    files_data, count, filtered = prepare_files_data(test_df, 'source', 'target', 'test')
    assert count == 2, 'Data preparation failed'
    assert len(files_data) == 5, 'Expected 5 output files'
    print('✓ Data preparation works')
    
    # Тест создания инструкций
    instructions = create_instructions('test', 2, 'source', 'target', True)
    assert 'test_bilingual_full.csv' in instructions, 'Instructions generation failed'
    print('✓ Instructions generation works')
    
    print()
    print('🎉 All tests passed! File Preparation App is ready to use.')
    
except Exception as e:
    print(f'❌ Test failed: {str(e)}')
    import traceback
    traceback.print_exc()
    exit(1)
"

if %errorlevel% neq 0 (
    echo.
    echo ❌ Tests failed! Please check the error messages above.
    echo.
    echo Possible solutions:
    echo 1. Run: setup_file_preparation.bat
    echo 2. Check if file_preparation_app.py exists
    echo 3. Verify Python installation
    pause
    exit /b 1
)

echo.
echo ===================================
echo Testing completed successfully!
echo ===================================
echo.
echo File Preparation App is ready to use.
echo.
echo To run the app:
echo   run_file_preparation.bat
echo.
echo Or from main menu:
echo   start.bat → option 5
echo.
pause
