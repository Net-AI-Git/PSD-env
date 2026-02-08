@echo off
setlocal enabledelayedexpansion
REM ===================================================================
REM PSD Analysis Tool - Download Wheels Script
REM ===================================================================
REM Downloads all required wheel files to install/wheels/ for offline installation
REM ===================================================================

echo.
echo ========================================
echo PSD Analysis Tool - Download Wheels
echo ========================================
echo.

REM ===== Step 1: Navigate to Project Root =====
echo [Step 1/4] Navigating to project directory...
cd /d "%~dp0.."
if errorlevel 1 (
    echo [ERROR] Failed to change to project directory!
    pause
    exit /b 1
)
echo [OK] Project directory: %CD%
echo.

REM ===== Step 2: Check Python 3.12 =====
echo [Step 2/4] Checking for Python 3.12...
set PYTHON_CMD=

REM Try to find Python 3.12 using py launcher (Windows)
py -3.12 --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=py -3.12
    goto :python_found
)

REM Try python3.12 command
python3.12 --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=python3.12
    goto :python_found
)

REM Try python command and check version (supports 3.12.x like 3.12.6)
python --version >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
    echo %PYTHON_VERSION% | findstr /R "^3\.12" >nul
    if not errorlevel 1 (
        set PYTHON_CMD=python
        goto :python_found
    )
)

REM Python 3.12 not found
echo [ERROR] Python 3.12 not found!
echo.
echo Please install Python 3.12 or use: py -3.12
echo.
pause
exit /b 1

:python_found
REM Verify Python version
for /f "tokens=2" %%i in ('!PYTHON_CMD! --version 2^>^&1') do set PYTHON_VERSION=%%i
echo %PYTHON_VERSION% | findstr /R "^3\.12" >nul
if errorlevel 1 (
    echo [ERROR] Python version verification failed!
    pause
    exit /b 1
)
echo [OK] Python version: %PYTHON_VERSION%
echo.

REM ===== Step 3: Check Requirements =====
echo [Step 3/4] Checking requirements.txt...
if not exist "requirements.txt" (
    echo [ERROR] requirements.txt not found!
    pause
    exit /b 1
)
echo [OK] requirements.txt found
echo.

REM ===== Step 4: Prepare Wheels Directory =====
echo [Step 4/4] Preparing wheels directory...
if not exist "install\wheels" (
    mkdir "install\wheels"
)
echo [OK] Wheels directory ready
echo.

REM ===== Download Wheels =====
echo ========================================
echo Downloading wheels for Python 3.12...
echo ========================================
echo.
echo Packages to download:
for /f "usebackq tokens=* delims=" %%p in ("requirements.txt") do (
    for /f "tokens=1 delims=><=!" %%a in ("%%p") do echo   - %%a
)
echo.
echo Target: %CD%\install\wheels
echo.
set /p CONFIRM="Continue? (y/n): "
if /i not "!CONFIRM!"=="y" (
    echo Download cancelled.
    pause
    exit /b 0
)
echo.

echo Starting download...
!PYTHON_CMD! -m pip download -r requirements.txt -d install/wheels --only-binary :all: --no-cache-dir
if errorlevel 1 (
    echo.
    echo [ERROR] Download failed!
    pause
    exit /b 1
)

REM Count downloaded wheels
for /f %%i in ('dir /b "install\wheels\*.whl" 2^>nul ^| find /c /v ""') do set WHEEL_COUNT=%%i
if not defined WHEEL_COUNT set WHEEL_COUNT=0

if !WHEEL_COUNT!==0 (
    echo.
    echo [ERROR] No wheel files downloaded!
    pause
    exit /b 1
)

echo.
echo ========================================
echo Download Complete!
echo ========================================
echo.
echo Total wheel files: !WHEEL_COUNT!
echo Location: %CD%\install\wheels
echo.
echo You can now run install\setup.bat to install offline.
echo.
pause
