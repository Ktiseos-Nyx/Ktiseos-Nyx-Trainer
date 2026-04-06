@echo off
title Ktiseos-Nyx-Trainer Diagnostics

echo.
echo ======================================================================
echo Ktiseos-Nyx-Trainer Diagnostic Tool
echo ======================================================================
echo.
echo This tool will collect system information to help troubleshoot
echo installation issues. It will create two files:
echo   - diagnostics_TIMESTAMP.json (for developers)
echo   - diagnostics_TIMESTAMP.txt (human-readable report)
echo.
echo ======================================================================
echo.

REM Use %~dp0 so paths work regardless of CWD (e.g. project on I:\ drive)
set SCRIPT_DIR=%~dp0

REM Prefer venv Python directly — avoids activation fragility and ensures
REM sys.executable points to the venv so package checks find torch, etc.
if exist "%SCRIPT_DIR%.venv\Scripts\python.exe" (
    set PYTHON_CMD="%SCRIPT_DIR%.venv\Scripts\python.exe"
    echo Using venv Python: %SCRIPT_DIR%.venv\Scripts\python.exe
    goto :RUN_DIAGNOSTIC
)

REM No venv — fall back to system Python
REM Try py launcher first
where py >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=py -3
    echo Using system Python via py launcher
    goto :RUN_DIAGNOSTIC
)

REM Try python3
where python3 >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=python3
    echo Using system python3
    goto :RUN_DIAGNOSTIC
)

REM Try python
where python >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=python
    echo Using system python
    goto :RUN_DIAGNOSTIC
)

echo ERROR: Python not found!
echo.
echo Please install Python 3.10+ from https://python.org/downloads/
echo Make sure to check "Add Python to PATH" during installation.
echo.
pause
exit /b 1

:RUN_DIAGNOSTIC

echo.

%PYTHON_CMD% "%SCRIPT_DIR%diagnose.py"

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Diagnostic script failed.
    echo.
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo Diagnostic files created successfully!
echo ======================================================================
echo.
echo Please attach the logs\diagnostics_*.txt file to your GitHub issue.
echo.
pause
