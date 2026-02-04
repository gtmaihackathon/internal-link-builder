@echo off
REM ============================================
REM Internal Link Builder - Windows Setup Script
REM ============================================

echo.
echo ============================================
echo    Internal Link Builder - Setup Script
echo ============================================
echo.

REM Check Python
echo [INFO] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://www.python.org/downloads/
    pause
    exit /b 1
)
echo [SUCCESS] Python found

REM Create directories
echo [INFO] Creating directory structure...
if not exist "src" mkdir src
if not exist "config" mkdir config
if not exist "tests" mkdir tests
if not exist "data" mkdir data
if not exist "exports" mkdir exports
if not exist ".github\workflows" mkdir .github\workflows

REM Create placeholder files
echo. > src\__init__.py
echo. > config\__init__.py
echo. > tests\__init__.py
echo. > data\.gitkeep
echo. > exports\.gitkeep

echo [SUCCESS] Directory structure created

REM Create virtual environment
echo [INFO] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo [SUCCESS] Virtual environment created
) else (
    echo [WARNING] Virtual environment already exists
)

REM Activate and install
echo [INFO] Installing dependencies...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet
if exist "requirements.txt" (
    pip install -r requirements.txt --quiet
    echo [SUCCESS] Dependencies installed
) else (
    echo [WARNING] requirements.txt not found
)

REM Move source files if needed
if exist "app.py" if not exist "src\app.py" (
    move app.py src\ >nul
    echo [SUCCESS] Moved app.py to src\
)
if exist "cli.py" if not exist "src\cli.py" (
    move cli.py src\ >nul
    echo [SUCCESS] Moved cli.py to src\
)
if exist "api.py" if not exist "src\api.py" (
    move api.py src\ >nul
    echo [SUCCESS] Moved api.py to src\
)

REM Create run scripts
echo [INFO] Creating run scripts...

echo @echo off > run_app.bat
echo call venv\Scripts\activate.bat >> run_app.bat
echo streamlit run src\app.py --server.port 8501 >> run_app.bat

echo @echo off > run_api.bat
echo call venv\Scripts\activate.bat >> run_api.bat
echo uvicorn src.api:app --reload --port 8000 >> run_api.bat

echo [SUCCESS] Created run scripts

echo.
echo ============================================
echo    Setup Complete!
echo ============================================
echo.
echo Next steps:
echo.
echo 1. Run Streamlit app:
echo    run_app.bat
echo.
echo 2. Run API server:
echo    run_api.bat
echo.
echo 3. Use CLI:
echo    venv\Scripts\activate.bat
echo    python src\cli.py crawl urls.txt --user-agent googlebot
echo.
echo Happy internal linking!
echo.
pause
