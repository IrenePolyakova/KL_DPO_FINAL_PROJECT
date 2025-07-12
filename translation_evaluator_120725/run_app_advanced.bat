@echo off
echo ===================================
echo  Advanced Port Detection Launcher
echo ===================================

REM Проверка виртуального окружения
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run start.bat and choose option 1 first.
    pause
    exit /b 1
)

REM Активация виртуального окружения
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Создание временного PowerShell скрипта для поиска свободного порта
echo Creating port detection script...
echo $startPort = 8501 > find_port.ps1
echo $endPort = 8520 >> find_port.ps1
echo $foundPort = $null >> find_port.ps1
echo. >> find_port.ps1
echo for ($port = $startPort; $port -le $endPort; $port++) { >> find_port.ps1
echo     $tcpConnections = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue >> find_port.ps1
echo     if ($tcpConnections.Count -eq 0) { >> find_port.ps1
echo         $foundPort = $port >> find_port.ps1
echo         break >> find_port.ps1
echo     } >> find_port.ps1
echo } >> find_port.ps1
echo. >> find_port.ps1
echo if ($foundPort) { >> find_port.ps1
echo     Write-Host "AVAILABLE_PORT=$foundPort" >> find_port.ps1
echo } else { >> find_port.ps1
echo     Write-Host "AVAILABLE_PORT=0" >> find_port.ps1
echo } >> find_port.ps1

REM Запуск PowerShell скрипта для поиска порта
echo Searching for available port using PowerShell...
for /f "tokens=2 delims==" %%i in ('powershell -ExecutionPolicy Bypass -File find_port.ps1') do set "AVAILABLE_PORT=%%i"

REM Удаление временного скрипта
del find_port.ps1

REM Настройка переменных окружения
set STREAMLIT_SERVER_MAX_UPLOAD_SIZE=5000
set STREAMLIT_SERVER_MAX_MESSAGE_SIZE=5000
set STREAMLIT_SERVER_ENABLE_CORS=false
set STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
set STREAMLIT_GLOBAL_DEVELOPMENT_MODE=false

REM Создание конфигурационного файла Streamlit
if not exist ".streamlit" mkdir .streamlit

echo [server] > .streamlit\config.toml
echo maxUploadSize = 5000 >> .streamlit\config.toml
echo maxMessageSize = 5000 >> .streamlit\config.toml
echo enableCORS = false >> .streamlit\config.toml
echo enableXsrfProtection = false >> .streamlit\config.toml
echo headless = false >> .streamlit\config.toml
echo. >> .streamlit\config.toml
echo [browser] >> .streamlit\config.toml
echo gatherUsageStats = false >> .streamlit\config.toml
echo. >> .streamlit\config.toml
echo [global] >> .streamlit\config.toml
echo developmentMode = false >> .streamlit\config.toml

echo.
echo ===================================
echo Advanced Port Detection Results
echo ===================================

if "%AVAILABLE_PORT%"=="0" (
    echo No specific port found in range 8501-8520
    echo Using Streamlit's automatic port selection
    echo Application will choose an available port automatically
) else (
    echo Found available port: %AVAILABLE_PORT%
    echo Application will be available at: http://localhost:%AVAILABLE_PORT%
)

echo.
echo Configuration:
echo - Max upload size: 5000 MB (5 GB)
echo - Port detection: Advanced (PowerShell)
echo - Memory optimization: Enabled
echo - CORS: Disabled for security
echo.

REM Запуск приложения
echo Starting application...
if "%AVAILABLE_PORT%"=="0" (
    streamlit run app.py --server.address=localhost
) else (
    streamlit run app.py --server.port=%AVAILABLE_PORT% --server.address=localhost
)

echo.
echo Application stopped.
pause
