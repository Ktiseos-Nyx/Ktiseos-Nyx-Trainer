@echo off
title Ktiseos Nyx Trainer Installer

REM Set PYTHONIOENCODING to UTF-8 for consistent output on Windows
set PYTHONIOENCODING=utf-8

echo Running Ktiseos Nyx Trainer Installer for Windows...
echo.

REM Find the best available python command (prioritize newer versions)
set "PYTHON_CMD=python"
where python3.12 >nul 2>&1 && set "PYTHON_CMD=python3.12"
where python3.11 >nul 2>&1 && set "PYTHON_CMD=python3.11"
where python3.10 >nul 2>&1 && set "PYTHON_CMD=python3.10"
where python3 >nul 2>&1 && set "PYTHON_CMD=python3"

echo Using Python command: %PYTHON_CMD%
echo    If this is incorrect, please edit this script or ensure the correct Python is in your PATH.
echo.

REM Verify Python is >= 3.10
for /f "tokens=2 delims=. " %%i in ('%PYTHON_CMD% --version 2^>^&1') do (
    if %%i LSS 10 (
        echo Python 3.10+ required. Found version %PYTHON_CMD%.
        pause
        exit /b 1
    )
)

REM Execute the local Windows installer script
"%PYTHON_CMD%" installer_windows_local.py %*

if %errorlevel% neq 0 (
    echo.
    echo Installation failed with an error.
    echo    Please review the messages above.
    pause
) else (
    echo.
    echo Installation completed successfully!
    echo.
    echo Next steps:
    echo   - Local development: start_services_local.bat
    echo   - VastAI deployment: Services auto-start via supervisor
    echo.
    pause
)
