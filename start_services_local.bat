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
SET FRONTEND_PORT=3000
SET BACKEND_PORT=8000

REM --- Parse arguments ---
REM Usage: start_services_local.bat [--port 3000] [--backend-port 8000]
:parse_args
if "%~1"=="" goto :args_done
if /I "%~1"=="--port" (
    if "%~2"=="" (
        echo [ERROR] --port requires a port number.
        pause
        exit /b 1
    )
    SET FRONTEND_PORT=%~2
    shift
    shift
    goto :parse_args
)
if /I "%~1"=="--backend-port" (
    if "%~2"=="" (
        echo [ERROR] --backend-port requires a port number.
        pause
        exit /b 1
    )
    SET BACKEND_PORT=%~2
    shift
    shift
    goto :parse_args
)
if /I "%~1"=="-h" goto :show_help
if /I "%~1"=="--help" goto :show_help
echo [ERROR] Unknown argument: %~1
echo Run '%~nx0 --help' for usage.
pause
exit /b 1

:show_help
echo Usage: %~nx0 [--port FRONTEND_PORT] [--backend-port BACKEND_PORT]
echo.
echo   --port           Frontend port (default: 3000)
echo   --backend-port   Backend API port (default: 8000)
echo.
echo Example: %~nx0 --port 4000 --backend-port 9000
exit /b 0

:args_done

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

REM --- Check: Ports already in use ---
netstat -ano 2>nul | findstr ":!BACKEND_PORT! " | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo [WARNING] Port !BACKEND_PORT! is already in use!
    echo           Something is already listening there. The backend may fail to start.
    echo           To find what's using it: netstat -ano ^| findstr ":!BACKEND_PORT!"
    echo           Tip: use --backend-port to pick a different port.
    echo.
    SET HAS_WARNINGS=1
)

netstat -ano 2>nul | findstr ":!FRONTEND_PORT! " | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo [WARNING] Port !FRONTEND_PORT! is already in use!
    echo           Something is already listening there. The frontend may fail to start.
    echo           To find what's using it: netstat -ano ^| findstr ":!FRONTEND_PORT!"
    echo           Tip: use --port to pick a different port.
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
    echo [Backend] Starting FastAPI backend on http://localhost:!BACKEND_PORT!...
    start "Ktiseos Backend" /MIN %PYTHON_EXE% -m uvicorn api.main:app --host 127.0.0.1 --port !BACKEND_PORT! --reload
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
        echo [Frontend] Starting Next.js frontend on http://localhost:!FRONTEND_PORT!...
        start "Ktiseos Frontend" /MIN cmd /c "cd frontend && set NODE_ENV=production&& set PORT=!FRONTEND_PORT!&& set BACKEND_PORT=!BACKEND_PORT!&& npm start"
    )
)

echo.
echo ==========================================
echo [SUCCESS] Local Services Started!
echo ==========================================
echo.
echo   Access the UI at: http://localhost:!FRONTEND_PORT!
echo   API Docs available at: http://localhost:!BACKEND_PORT!/docs
echo.
echo [INFO] To stop services, close the minimized command windows
echo        or press Ctrl+C in each one.
echo.
