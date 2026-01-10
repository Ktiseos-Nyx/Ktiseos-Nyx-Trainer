@echo off
title Ktiseos Nyx Trainer Installer

REM ======================================================================
REM Windows Installer for Ktiseos-Nyx-Trainer
REM Supports interactive venv creation and flag-based automation
REM ======================================================================

set PYTHONIOENCODING=utf-8
set USE_VENV=
set VENV_DIR=venv
set AUTO_MODE=0

REM Parse command-line arguments
:PARSE_ARGS
if "%~1"=="" goto :ARGS_DONE
if /I "%~1"=="--venv" (
    set USE_VENV=1
    shift
    goto :PARSE_ARGS
)
if /I "%~1"=="--no-venv" (
    set USE_VENV=0
    shift
    goto :PARSE_ARGS
)
if /I "%~1"=="--auto" (
    set USE_VENV=1
    set AUTO_MODE=1
    shift
    goto :PARSE_ARGS
)
REM Keep other arguments like --verbose
shift
goto :PARSE_ARGS
:ARGS_DONE

echo ======================================================================
echo Ktiseos Nyx Trainer - Windows Installer
echo ======================================================================
echo.

REM If no venv flag was set, ask the user interactively
if not defined USE_VENV (
    echo.
    echo Virtual Environment Recommendation:
    echo ======================================================================
    echo A virtual environment (venv) isolates this project's packages from
    echo other Python projects on your system. This prevents version conflicts
    echo and makes troubleshooting much easier.
    echo.
    echo Benefits:
    echo   - No conflicts with other Python projects
    echo   - Easy to delete if you want to start fresh (just delete 'venv' folder^)
    echo   - Industry best practice for Python development
    echo.
    echo The venv will be created in: %CD%\%VENV_DIR%
    echo ======================================================================
    echo.
    choice /C YN /M "Create a virtual environment? (Recommended: Y)"
    if errorlevel 2 (
        set USE_VENV=0
        echo.
        echo Proceeding WITHOUT virtual environment...
        echo (You can always re-run with --venv flag later^)
        echo.
    ) else (
        set USE_VENV=1
        echo.
        echo Creating virtual environment...
        echo.
    )
)

REM Create venv if requested
if "%USE_VENV%"=="1" (
    if not exist "%VENV_DIR%\Scripts\activate.bat" (
        echo Creating virtual environment in '%VENV_DIR%'...
        py -3.10 -m venv %VENV_DIR%
        if errorlevel 1 (
            echo.
            echo ERROR: Failed to create virtual environment.
            echo Make sure Python 3.10+ is installed from python.org
            pause
            exit /b 1
        )
        echo Virtual environment created successfully!
        echo.
    ) else (
        echo Virtual environment already exists.
        echo.
    )

    echo Activating virtual environment...
    call %VENV_DIR%\Scripts\activate.bat
    echo.
    echo ^(venv^) Virtual environment activated.
    echo Python: %VENV_DIR%\Scripts\python.exe
    echo.
    echo Running installer inside venv...
    echo.

    REM Run installer with venv Python, passing through remaining args
    %VENV_DIR%\Scripts\python.exe installer_windows_local.py %*
) else (
    echo Running installer WITHOUT virtual environment...
    echo.
    echo Checking Python installation...
    echo.

    REM Use py.exe launcher to find Python 3.10+
    py -3.10 installer_windows_local.py %*
)

REM Check the exit code from the Python script
if %errorlevel% neq 0 (
    echo.
    echo ======================================================================
    echo ERROR: The installer failed!
    echo ======================================================================
    echo.
    echo Common issues:
    echo   1. Python 3.10+ not installed
    echo      Solution: Install from https://python.org/downloads/
    echo      During install, CHECK "Add Python to PATH"
    echo.
    echo   2. Microsoft Store Python detected
    echo      Solution: Uninstall MS Store Python, install from python.org
    echo.
    echo   3. Multiple Python installations conflicting
    echo      Solution: Check "python --version" and "py --list" in CMD
    echo.
    echo Check the log file in logs/ folder for detailed error information.
    echo ======================================================================
) else (
    echo.
    echo ======================================================================
    echo Installation completed successfully!
    echo ======================================================================
    echo.
    echo IMPORTANT: Review any warnings above about:
    echo   - Microsoft Store Python
    echo   - CPU-only PyTorch ^(GPU training won't work^)
    echo.
    echo Next steps:
    echo   1. Ensure PyTorch has CUDA support ^(see warnings above^)
    echo   2. Run 'start_services_local.bat' to start the web UI
    echo   3. Access at http://localhost:3000
    echo ======================================================================
)

echo.
echo Press any key to close this window...
pause > nul