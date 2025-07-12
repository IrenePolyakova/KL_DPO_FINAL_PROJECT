@echo off
echo ===================================
echo  System Cleanup and Optimization
echo ===================================

REM Очистка кэша Python
echo Cleaning Python cache...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
for /r . %%f in (*.pyc) do @if exist "%%f" del /q "%%f"
for /r . %%f in (*.pyo) do @if exist "%%f" del /q "%%f"

REM Очистка кэша моделей и загрузок
echo Cleaning PyTorch cache...
if exist "%USERPROFILE%\.cache\torch" (
    rd /s /q "%USERPROFILE%\.cache\torch" 2>nul
)
if exist "torch_cache" (
    rd /s /q "torch_cache" 2>nul
    mkdir torch_cache
)

echo Cleaning Transformers cache...
if exist "%USERPROFILE%\.cache\huggingface" (
    rd /s /q "%USERPROFILE%\.cache\huggingface\transformers" 2>nul
)
if exist "transformers_cache" (
    rd /s /q "transformers_cache" 2>nul
    mkdir transformers_cache
)

echo Cleaning HuggingFace cache...
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
        echo Cleaning Streamlit logs...
        rd /s /q ".streamlit\logs"
    )
)

REM Очистка временных файлов Windows
if exist "%TEMP%\streamlit*" (
    echo Cleaning Windows temp files...
    del /q "%TEMP%\streamlit*" 2>nul
)

echo.
echo ===================================
echo System Information
echo ===================================
echo.
echo Available disk space:
dir /-c | findstr /i "bytes free"
echo.

echo Memory usage:
wmic OS get TotalVisibleMemorySize,FreePhysicalMemory /value | findstr "="
echo.

echo CPU Information:
wmic CPU get Name,NumberOfCores,NumberOfLogicalProcessors /value | findstr "="
echo.

echo GPU Information:
wmic PATH Win32_VideoController get Name /value | findstr "=" | findstr /v "Microsoft"
echo.

echo Python Environment Status:
if exist "venv\Scripts\activate.bat" (
    echo ✓ Virtual environment found
    call venv\Scripts\activate.bat && python --version 2>nul && echo ✓ Python is working
) else (
    echo ✗ Virtual environment not found
)
echo.

REM Проверка размера файлов в проекте
echo Checking for large files in project...
echo Large files (models, data) should be in designated folders:
if exist "uploaded_finetuned_models" (
    echo [Models folder] Found model files:
    dir "uploaded_finetuned_models" | findstr /i ".zip"
)
if exist "data" (
    echo [Data folder] Found data files:
    dir "data" | findstr /i ".csv .docx .xlsx"
)
echo.

echo.
echo ===================================
echo Cleanup completed!
echo ===================================
echo.
echo Tips for better performance:
echo - Close unnecessary applications
echo - Ensure sufficient free disk space (at least 10GB)
echo - Use SSD storage if possible
echo - Consider increasing virtual memory if working with large models
echo.
pause
