@echo off
rem SubZeroAstro - launcher
rem First run: creates a private Python environment and installs PySide6 + NumPy (a few minutes).
rem Later runs: starts straight away.

cd /d "%~dp0"

if exist ".venv\Scripts\pythonw.exe" goto run

echo.
echo  SubZeroAstro - first-time setup
echo  -------------------------------------------
where py >nul 2>nul
if %errorlevel%==0 (
    py -3 -m venv .venv
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo.
        echo  Python was not found.
        echo  Install Python 3.11 or newer from https://www.python.org/downloads/
        echo  and tick "Add python.exe to PATH" during installation, then run this again.
        echo.
        pause
        exit /b 1
    )
    python -m venv .venv
)
if not exist ".venv\Scripts\python.exe" (
    echo  Could not create the Python environment.
    pause
    exit /b 1
)
echo  Installing PySide6 and NumPy...
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo  Installation failed - check your internet connection and try again.
    rmdir /s /q .venv
    pause
    exit /b 1
)
echo  Setup complete.

:run
start "" ".venv\Scripts\pythonw.exe" subzeroastro.py %*
