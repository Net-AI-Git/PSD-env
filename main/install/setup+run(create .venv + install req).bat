@echo on
setlocal

REM ===== Move to project root directory (one level up from install folder) =====
cd /d "%~dp0.."

REM ===== Check if virtual environment exists =====
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

REM ===== Activate the virtual environment =====
call .venv\Scripts\activate

REM ===== Install requirements =====
echo Installing required packages...
pip install --upgrade pip >nul
pip install -r requirements.txt

REM ===== Run the Bokeh GUI =====
echo Starting Bokeh server...
bokeh serve app/gui.py --show

REM ===== Keep window open after exit =====
pause
