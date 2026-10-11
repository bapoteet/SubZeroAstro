@echo off
rem Builds a stand-alone "FITS Blaster.exe" (no Python needed on the PC that runs it).
rem Run run.bat once first so the Python environment exists.

cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run run.bat once first to set up Python, then run this again.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m pip install pyinstaller
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean --windowed ^
    --name "FITS Blaster" fits_blaster.py
if errorlevel 1 (
    echo Build failed.
    pause
    exit /b 1
)
echo.
echo Done. Your app is in:  dist\FITS Blaster\FITS Blaster.exe
echo Copy the whole "dist\FITS Blaster" folder wherever you like.
pause
