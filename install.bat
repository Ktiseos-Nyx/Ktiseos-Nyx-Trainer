@echo off
title LoRA Easy Training Installer

echo 🚀 Running LoRA Easy Training Installer for Windows...
echo.

REM This script will execute the main Python installer.

REM Find the best available python command
set "PYTHON_CMD=python"
where python3.11 >nul 2>nul && set "PYTHON_CMD=python3.11"
where python3.10 >nul 2>nul && set "PYTHON_CMD=python3.10"

echo 🐍 Using Python command: %PYTHON_CMD%
echo    If this is incorrect, please edit this script or ensure the correct Python is in your PATH.
echo.

REM Execute the installer script
"%PYTHON_CMD%" installer.py

if %errorlevel% neq 0 (
    echo.
    echo ❌ Installation failed with an error.
    echo    Please review the messages above.
    pause
) else (
    echo.
    echo ✅ Installation seems to have completed successfully.
    echo    Run start_jupyter.bat to begin your session.
)
