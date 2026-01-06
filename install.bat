@echo off
title Ktiseos Nyx Trainer Installer

REM ======================================================================
REM This script uses the recommended 'py.exe' launcher for Windows to
REM reliably find and use the correct Python version, avoiding common
REM PATH and Microsoft Store alias issues.
REM ======================================================================

set PYTHONIOENCODING=utf-8

echo Running Ktiseos Nyx Trainer Installer for Windows...
echo.

REM Use the py.exe launcher to specifically request Python 3.10 or newer.
REM This is the most robust method and is the official standard.
REM It will fail gracefully if a suitable version is not found.
REM The '%*' passes along any command-line arguments (like --verbose).
py -3.10 installer_windows_local.py %*

REM Check the exit code from the Python script
if %errorlevel% neq 0 (
    echo.
    echo ----------------------------------------------------------------------
    echo ERROR: The installer failed.
    echo Please review the messages above. If the error mentions Python,
    echo ensure you have Python 3.10 or newer installed from python.org.
    echo ----------------------------------------------------------------------
) else (
    echo.
    echo ----------------------------------------------------------------------
    echo Installation completed successfully!
    echo.
    echo Next steps: Run 'start_services_local.bat' to start the application.
    echo ----------------------------------------------------------------------
)

echo.
echo Press any key to close this window.
pause > nul