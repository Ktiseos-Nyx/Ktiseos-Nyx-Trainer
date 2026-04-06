@echo off
title Ktiseos Nyx Trainer - Quick Restart

REM UTF-8 for consistent output
set PYTHONIOENCODING=utf-8

echo ==========================================
echo ⚡ Quick Restart (Skipping Dependency Installation)...
echo ==========================================

REM --------------------------------------------------------------------
REM Step 0: Clean up ports 8000 and 3000
REM --------------------------------------------------------------------
echo 🧹 Cleaning up existing processes...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do taskkill /F /PID %%a 2>nul
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :3000 ^| findstr LISTENING') do taskkill /F /PID %%a 2>nul
taskkill /F /IM python.exe /FI "WINDOWTITLE eq Ktiseos Backend*" 2>nul
taskkill /F /IM node.exe /FI "WINDOWTITLE eq Ktiseos Frontend*" 2>nul

timeout /t 2 /nobreak >nul

REM --------------------------------------------------------------------
REM Step 1: Run installer with --skip-install
REM --------------------------------------------------------------------
echo ⚙️ Verifying environment (skipping reinstall)...
call install.bat --skip-install

REM --------------------------------------------------------------------
REM Step 2: Start services (production mode)
REM --------------------------------------------------------------------
echo.
echo 🚀 Starting Services...
call start_services_local.bat
