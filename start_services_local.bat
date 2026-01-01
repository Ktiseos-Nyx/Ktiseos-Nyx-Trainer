@echo off
REM Service Startup Script for Ktiseos-Nyx-Trainer (Local Development - Windows)
REM This script starts both the FastAPI backend and Next.js frontend on Windows

title Ktiseos-Nyx-Trainer Local Services

echo ==========================================
echo 🚀 Starting Ktiseos-Nyx-Trainer Services (Local)...
echo ==========================================
echo.

REM --------------------------------------------------------------------
REM Step 0: Clean up any existing processes using the ports
REM --------------------------------------------------------------------
echo 🧹 Cleaning up any existing processes on ports 8000 and 3000...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do taskkill /F /PID %%a 2>nul
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :3000 ^| findstr LISTENING') do taskkill /F /PID %%a 2>nul

timeout /t 3 /nobreak >nul

REM --------------------------------------------------------------------
REM Step 1: Ensure environment is set up via Windows installer
REM --------------------------------------------------------------------
echo ⚙️ Running Windows local installer to set up environment...

REM Find the best available python command
set "PYTHON_CMD=python"
where python3 >nul 2>nul && set "PYTHON_CMD=python3"

"%PYTHON_CMD%" installer_windows_local.py
if %errorlevel% neq 0 (
    echo ❌ Installer failed! Please check the errors above.
    pause
    exit /b 1
)

REM --------------------------------------------------------------------
REM Step 2: Start Services
REM --------------------------------------------------------------------

REM Start FastAPI backend (if api directory exists)
if exist "api\" (
    echo 🐍 Starting FastAPI backend on port 8000 (bind 127.0.0.1)...
    start "Ktiseos Backend" /MIN "%PYTHON_CMD%" -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
    echo    Backend started in background window
) else (
    echo ⚠️  API directory not found - skipping backend startup
    echo    Create /api/main.py with FastAPI app
)

REM Start Next.js frontend (if frontend directory exists)
if exist "frontend\" (
    echo 🎨 Starting Next.js frontend on port 3000...

    cd frontend

    REM Check if node is available
    where node >nul 2>nul
    if %errorlevel% neq 0 (
        echo ❌ Node.js not found! Please install Node.js 18+ and try again.
        cd ..
        pause
        exit /b 1
    )

    REM FIX: Use PORT env var (Next.js standard)
    start "Ktiseos Frontend" /MIN cmd /c "set PORT=3000 && npm run start"
    echo    Frontend started in background window
    cd ..
) else (
    echo ⚠️  Frontend directory not found - skipping frontend startup
    echo    Create /frontend with Next.js app
)

echo.
echo ==========================================
echo ✅ Local Services Started!
echo ==========================================
echo.
echo 🌐 Access URLs:
echo    Frontend: http://localhost:3000
echo    Backend API: http://localhost:8000
echo    API Docs: http://localhost:8000/docs
echo.
echo 📊 Services running in minimized windows
echo.
echo 🛑 To stop services:
echo    - Close the minimized "Ktiseos Backend" and "Ktiseos Frontend" windows
echo    - Or run: taskkill /F /IM python.exe /FI "WINDOWTITLE eq Ktiseos Backend*"
echo    - Or run: taskkill /F /IM node.exe /FI "WINDOWTITLE eq Ktiseos Frontend*"
echo.
echo Press any key to exit this window (services will continue running)...
pause >nul
