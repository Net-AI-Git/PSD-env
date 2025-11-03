@echo off
setlocal
REM Batch file to start the PSD Analysis Tool Bokeh GUI

echo ========================================
echo PSD Analysis Tool
echo ========================================
echo.

REM Change to project root directory (one level up from install folder)
echo Changing to project directory...
cd /d "%~dp0.."
if errorlevel 1 (
    echo [ERROR] Failed to change to project directory!
    echo Current location: %~dp0
    pause
    exit /b 1
)

REM Show current directory
echo Current Directory:
cd
echo.

REM Check if app/gui.py exists
echo Checking for gui.py file...
if exist "app\gui.py" (
    echo [OK] Found: app\gui.py
) else (
    echo [ERROR] Cannot find: app\gui.py
    echo Please check that the file exists in the project directory
    pause
    exit /b 1
)
echo.

REM Check if virtual environment exists and activate it
if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
    echo [OK] Virtual environment activated
) else (
    echo [INFO] Virtual environment not found, using system Python
    echo [INFO] To create a virtual environment, run the setup batch file in the install folder
)
echo.

REM Check if Python is available
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not found in PATH!
    echo Please make sure Python is installed and added to your system PATH.
    echo.
    pause
    exit /b 1
)
python --version
echo.

REM Check if bokeh is installed
echo Checking Bokeh installation...
python -c "import bokeh; print('[OK] Bokeh version:', bokeh.__version__)" 2>nul
if errorlevel 1 (
    echo [ERROR] Bokeh is not installed!
    echo.
    echo Please install Bokeh with one of the following:
    echo   1. Run the setup batch file in the install folder
    echo   2. Or install manually: pip install bokeh
    echo.
    pause
    exit /b 1
)
echo.

echo ========================================
echo Starting Bokeh Server...
echo ========================================
echo.
echo The browser will open automatically at:
echo http://localhost:5006/gui
echo.
echo IMPORTANT: Keep this terminal window open!
echo - Optimization progress will be shown here
echo - Press Ctrl+C to stop the server
echo ========================================
echo.

REM Start the Bokeh server using Python from the virtual environment
REM Using --dev flag enables auto-reload during development (optional)
python -m bokeh serve app\gui.py --show --address localhost --port 5006

REM If the above fails, try alternative path format
if errorlevel 1 (
    echo.
    echo Retrying with alternative path format...
    python -m bokeh serve app/gui.py --show --address localhost --port 5006
)

echo.
echo ========================================
echo Server Stopped
echo ========================================
pause