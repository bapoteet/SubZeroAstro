@echo off
rem Builds a stand-alone "SubZeroAstro.exe" (no Python needed on the PC that runs it).
rem Run run.bat once first so the Python environment exists.

cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run run.bat once first to set up Python, then run this again.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m pip install pyinstaller
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean --windowed ^
    --name "SubZeroAstro" subzeroastro.py
if errorlevel 1 (
    echo Build failed.
    pause
    exit /b 1
)
echo.
echo Done. Your app is in:  dist\SubZeroAstro\SubZeroAstro.exe
echo Copy the whole "dist\SubZeroAstro" folder wherever you like.
pause
