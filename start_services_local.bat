@echo off
title Ktiseos-Nyx-Trainer Local Services

REM ======================================================================
REM This script STARTS the services. It assumes the one-time installer
REM ('install_windows_local.bat') has already been run successfully.
REM
REM It correctly uses the Python executable from the virtual environment (.venv)
REM to ensure all dependencies are found and services run correctly.
REM ======================================================================

SETLOCAL

REM Define the virtual environment directory
SET VENV_DIR=%~dp0.venv

echo ==========================================
echo Starting Ktiseos-Nyx-Trainer Services...
echo ==========================================
echo.

REM --------------------------------------------------------------------
REM Step 1: Verify that the environment has been installed
REM --------------------------------------------------------------------
echo [Verifying installation...]
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found!
    echo    It looks like the installation hasn't been run yet.
    echo    Please run 'install_windows_local.bat' once before starting services.
    echo.
    pause
    exit /b 1
)

REM Set the Python command to the one INSIDE our venv for reliability
SET PYTHON_EXE="%VENV_DIR%\Scripts\python.exe"
echo [OK] Environment found. Using Python from: %PYTHON_EXE%
echo.

REM --------------------------------------------------------------------
REM Step 2: Clean up any existing processes using the ports
REM --------------------------------------------------------------------
echo [Cleanup] Cleaning up any existing processes on ports 8000 and 3000...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do taskkill /F /PID %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :3000 ^| findstr LISTENING') do taskkill /F /PID %%a >nul 2>&1
timeout /t 2 /nobreak >nul

REM --------------------------------------------------------------------
REM Step 3: Start Services using the correct Python
REM --------------------------------------------------------------------

REM Start FastAPI backend
if exist "api\" (
    echo [Backend] Starting FastAPI backend on http://localhost:8000...
    start "Ktiseos Backend" /MIN %PYTHON_EXE% -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
)

REM Start Next.js frontend
if exist "frontend\" (
    echo [Frontend] Starting Next.js frontend on http://localhost:3000...
    start "Ktiseos Frontend" /MIN cmd /c "cd frontend && npm start"
)

echo.
echo ==========================================
echo [SUCCESS] Local Services Started!
echo ==========================================
echo.
echo >> Access the UI at: http://localhost:3000
echo >> API Docs available at: http://localhost:8000/docs
echo.
echo [INFO] To stop services, simply close the new minimized command windows.
echo.
echo This window will now close.
timeout /t 5 /nobreak