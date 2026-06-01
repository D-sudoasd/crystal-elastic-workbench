@echo off
setlocal

cd /d "%~dp0"
set "APP_NAME=Crystal Elastic Workbench"
set "QT_API=pyside6"

if /I "%~1"=="--check" (
    echo Checking %APP_NAME% launcher...
    echo Project directory: %cd%
    py -3.11 -c "from PySide6.QtCore import Qt; from crystal_elastic_workbench.gui import MainWindow; print('GUI import check ok')"
    exit /b 0
)

echo Starting %APP_NAME%...
py -3.11 -m crystal_elastic_workbench
if %errorlevel% neq 0 (
    echo.
    echo Failed to start %APP_NAME% with py -3.11.
    echo Trying python instead...
    python -m crystal_elastic_workbench
)

if %errorlevel% neq 0 (
    echo.
    echo %APP_NAME% exited with an error.
    echo Please check that dependencies are installed:
    echo   py -3.11 -m pip install -r requirements.txt
    echo.
    pause
)
