@echo off
title Ktiseos-Nyx-Trainer Local Services

REM ======================================================================
REM This script STARTS the services. It assumes the one-time installer
REM ('install.bat') has already been run successfully.
REM
REM It correctly uses the Python executable from the virtual environment (.venv)
REM to ensure all dependencies are found and services run correctly.
REM ======================================================================

SETLOCAL EnableDelayedExpansion

REM Define the virtual environment directory
SET VENV_DIR=%~dp0.venv
SET HAS_WARNINGS=0

echo ==========================================
echo Starting Ktiseos-Nyx-Trainer Services...
echo ==========================================
echo.

REM ====================================================================
REM  PRE-FLIGHT CHECKS - Catch common issues BEFORE they become cryptic
REM ====================================================================
echo [Pre-flight checks...]
echo.

REM --- Check: Cloud sync folder (OneDrive, Dropbox, Google Drive) ---
REM These lock files during sync and cause "Access is denied" on writes
echo %CD% | findstr /I "OneDrive" >nul 2>&1
if not errorlevel 1 (
    echo [WARNING] Project is inside a OneDrive folder!
    echo           OneDrive can lock files during sync, causing "Access is denied".
    echo           Recommendation: Move the project to a non-synced folder
    echo           ^(e.g. C:\Projects or D:\Dev^)
    echo.
    SET HAS_WARNINGS=1
)
echo %CD% | findstr /I "Dropbox" >nul 2>&1
if not errorlevel 1 (
    echo [WARNING] Project is inside a Dropbox folder!
    echo           Dropbox can lock files during sync, causing "Access is denied".
    echo           Recommendation: Move the project to a non-synced folder.
    echo.
    SET HAS_WARNINGS=1
)
echo %CD% | findstr /I "Google Drive" >nul 2>&1
if not errorlevel 1 (
    echo [WARNING] Project is inside a Google Drive folder!
    echo           Google Drive can lock files during sync, causing "Access is denied".
    echo           Recommendation: Move the project to a non-synced folder.
    echo.
    SET HAS_WARNINGS=1
)

REM --- Check: Write permissions (can we actually write here?) ---
copy /y nul "%~dp0_write_test_.tmp" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Cannot write to project folder!
    echo         Path: %~dp0
    echo.
    echo         This usually means:
    echo           - The folder is read-only or controlled by another program
    echo           - You need to run as Administrator for this location
    echo           - The folder is on a read-only network drive
    echo.
    echo         Fix: Move the project to a folder you own, like:
    echo              C:\Users\%USERNAME%\Projects\Ktiseos-Nyx-Trainer
    echo.
    echo         DO NOT install to C:\ root, Program Files, or Windows folders.
    echo.
    echo         Run 'diagnose.bat' to collect system info for a bug report.
    echo.
	echo.		 DO NOT SUMMON EMET SELCH - Please For the Love of God.	
    pause
    exit /b 1
) else (
    del "%~dp0_write_test_.tmp" >nul 2>&1
)
)

REM --- Check: Ports already in use ---
netstat -ano 2>nul | findstr ":8000 " | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo [WARNING] Port 8000 is already in use!
    echo           Something is already listening there. The backend may fail to start.
    echo           To find what's using it: netstat -ano ^| findstr ":8000"
    echo.
    SET HAS_WARNINGS=1
)

netstat -ano 2>nul | findstr ":3000 " | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo [WARNING] Port 3000 is already in use!
    echo           Something is already listening there. The frontend may fail to start.
    echo           To find what's using it: netstat -ano ^| findstr ":3000"
    echo.
    SET HAS_WARNINGS=1
)

REM --- Check: Stale node processes (leftover from crash) ---
tasklist /FI "IMAGENAME eq node.exe" 2>nul | findstr /I "node.exe" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Node.js processes are already running.
    echo        If you're restarting, old processes may hold file locks.
    echo        If the frontend fails with "Access is denied" or "EBUSY",
    echo        close all Node terminals or run: taskkill /F /IM node.exe
    echo.
)

if "%HAS_WARNINGS%"=="1" (
    echo ------------------------------------------
    echo   Warnings found. Services may still work,
    echo   but check above if you hit errors.
    echo ------------------------------------------
    echo.
)

echo [Pre-flight checks complete.]
echo.

REM ====================================================================
REM Step 1: Verify that the environment has been installed
REM ====================================================================
echo [Verifying installation...]
if exist "%VENV_DIR%\Scripts\python.exe" (
    REM Virtual environment exists - use it
    SET PYTHON_EXE="%VENV_DIR%\Scripts\python.exe"
    echo [OK] Virtual environment found. Using Python from: %PYTHON_EXE%
) else (
    REM No venv - check if system Python is available
    where py >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Python not found!
        echo    Please run 'install.bat' to set up the environment first.
        echo.
        pause
        exit /b 1
    )
    REM Use system Python
    SET PYTHON_EXE=py -3
    echo [OK] Using system Python (no venv detected^)
)
echo.

REM ====================================================================
REM Step 2: Start Services
REM ====================================================================

REM Start FastAPI backend
if exist "api\" (
    echo [Backend] Starting FastAPI backend on http://localhost:8000...
    start "Ktiseos Backend" /MIN %PYTHON_EXE% -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
)

REM Start Next.js frontend
if exist "frontend\" (
    REM Check if npm is available
    where npm >nul 2>&1
    if errorlevel 1 (
        echo [Warning] npm not found - skipping frontend startup.
        echo            Install Node.js 20+ from https://nodejs.org to enable frontend.
    ) else (
        REM Check if node_modules exists
        if not exist "frontend\node_modules\next" (
            echo [Frontend] Dependencies missing, running npm install...
            pushd frontend
            npm install --legacy-peer-deps
            popd
        )
        REM Check if .next build exists
        if not exist "frontend\.next" (
            echo [Frontend] No build found, running npm run build...
            pushd frontend
            npm run build
            popd
        )
        echo [Frontend] Starting Next.js frontend on http://localhost:3000...
        start "Ktiseos Frontend" /MIN cmd /c "cd frontend && npm start"
    )
)

echo.
echo ==========================================
echo [SUCCESS] Local Services Started!
echo ==========================================
echo.
echo   Access the UI at: http://localhost:3000
echo   API Docs available at: http://localhost:8000/docs
echo.
echo [INFO] To stop services, close the minimized command windows
echo        or press Ctrl+C in each one.
echo.
