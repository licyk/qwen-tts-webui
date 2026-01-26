@echo off

python -V >NUL
if %ERRORLEVEL% NEQ 0 (
    echo Python not found, exit
    pause
    exit /b
)

python -c "import venv" >NUL
if %ERRORLEVEL% NEQ 0 (
    echo Venv module not found, exit
    pause
    exit /b
)

cd /d "%~dp0"

if not exist .\venv\Scripts\activate.bat (
    echo Create Venv...
    python -m venv venv
)

call .\venv\Scripts\activate.bat

python launch.py %*

pause