@echo off
setlocal enabledelayedexpansion
REM ===================================================================
REM PSD Analysis Tool - Installation Script
REM ===================================================================
REM Installs virtual environment and all packages from local wheels
REM ===================================================================

echo.
echo ========================================
echo PSD Analysis Tool - Installation
echo ========================================
echo.

REM ===== Step 1: Navigate to Project Root =====
echo [Step 1/5] Navigating to project directory...
cd /d "%~dp0.."
if errorlevel 1 (
    echo [ERROR] Failed to change to project directory!
    pause
    exit /b 1
)
echo [OK] Project directory: %CD%
echo.

REM ===== Step 2: Check Python 3.12 =====
echo [Step 2/5] Checking Python installation...
set PYTHON_CMD=
set PYTHON_FOUND=0

REM Try to find Python 3.12 using py launcher
py -3.12 --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=py -3.12
    for /f "tokens=2" %%i in ('py -3.12 --version 2^>^&1') do set PYTHON_VERSION=%%i
    set PYTHON_FOUND=1
    goto :python_check_done
)

REM Try python3.12 command
python3.12 --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=python3.12
    for /f "tokens=2" %%i in ('python3.12 --version 2^>^&1') do set PYTHON_VERSION=%%i
    set PYTHON_FOUND=1
    goto :python_check_done
)

REM Try python command and check if it's 3.12.x
python --version >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
    echo %PYTHON_VERSION% | findstr /R "^3\.12" >nul
    if not errorlevel 1 (
        set PYTHON_CMD=python
        set PYTHON_FOUND=1
        goto :python_check_done
    )
)

REM If no 3.12 found, try to find 3.12.6 specifically
echo [INFO] Python 3.12 not found, checking for Python 3.12.6...
py -3.12.6 --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=py -3.12.6
    for /f "tokens=2" %%i in ('py -3.12.6 --version 2^>^&1') do set PYTHON_VERSION=%%i
    set PYTHON_FOUND=1
    goto :python_check_done
)

python3.12.6 --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=python3.12.6
    for /f "tokens=2" %%i in ('python3.12.6 --version 2^>^&1') do set PYTHON_VERSION=%%i
    set PYTHON_FOUND=1
    goto :python_check_done
)

REM Python 3.12 or 3.12.6 not found
echo [ERROR] Python 3.12 or 3.12.6 not found!
echo.
echo Please install Python 3.12 or 3.12.6
pause
exit /b 1

:python_check_done
REM Verify it's actually 3.12.x
echo %PYTHON_VERSION% | findstr /R "^3\.12" >nul
if errorlevel 1 (
    echo [ERROR] Python version verification failed!
    pause
    exit /b 1
)
echo [OK] Python version: %PYTHON_VERSION%
echo.

REM ===== Step 3: Check Requirements and Wheels =====
echo [Step 3/5] Checking requirements and wheels...
if not exist "requirements.txt" (
    echo [ERROR] requirements.txt not found!
    pause
    exit /b 1
)

if not exist "install\wheels" (
    echo [ERROR] install\wheels directory not found!
    pause
    exit /b 1
)

dir /b "install\wheels\*.whl" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] No wheel files found in install\wheels!
    pause
    exit /b 1
)
echo [OK] Requirements and wheels found
echo.

REM ===== Step 4: Create Virtual Environment =====
echo [Step 4/5] Creating virtual environment...
if exist ".venv" (
    echo [INFO] Removing existing virtual environment...
    rmdir /s /q ".venv"
)

!PYTHON_CMD! -m venv .venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment!
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment!
    pause
    exit /b 1
)
echo [OK] Virtual environment created and activated
echo.

REM ===== Step 5: Install Packages =====
echo [Step 5/5] Installing packages from local wheels...
echo.

REM Count total packages to install
set PACKAGE_COUNT=0
for /f "usebackq tokens=* delims=" %%p in ("requirements.txt") do (
    set /a PACKAGE_COUNT+=1
)

echo Installing !PACKAGE_COUNT! package(s) and their dependencies...
echo This may take several minutes. Please wait...
echo.
echo Installing packages (pip will show progress below):
echo ----------------------------------------

REM Install packages - pip will show progress automatically
pip install --no-index --find-links install/wheels -r requirements.txt --progress-bar on
if errorlevel 1 (
    echo.
    echo [ERROR] Package installation failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo [OK] All packages installed successfully
echo ========================================
echo.

REM ===== Create Required Directories =====
echo Creating required directories...
if not exist "input" mkdir "input"
if not exist "results" mkdir "results"
echo [OK] Required directories ready
echo.

REM ===== Installation Complete =====
echo.
echo ========================================
echo ========================================
echo   INSTALLATION COMPLETE!
echo ========================================
echo ========================================
echo.
echo All packages have been installed successfully.
echo Virtual environment is ready.
echo.
echo To run the application:
echo   1. Navigate to: install folder
echo   2. Double-click: run_gui.bat
echo.
echo ========================================
echo Press any key to close this window...
pause >nul
