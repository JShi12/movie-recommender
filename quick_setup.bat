@echo off
REM Quick Setup - Try This First!

echo ============================================================
echo QUICK SETUP - TFX Recommender
echo ============================================================
echo.

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYVER=%%i
echo Your Python version: %PYVER%
echo.

REM Determine which setup to use
echo Detecting best setup for your Python version...
echo.

REM Extract major.minor version
for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do (
    set PYMAJOR=%%a
    set PYMINOR=%%b
)

if "%PYMAJOR%%PYMINOR%"=="39" goto FULL_TFX
if "%PYMAJOR%%PYMINOR%"=="310" goto FULL_TFX
goto MINIMAL

:FULL_TFX
echo ✅ Python 3.9 or 3.10 detected - Installing FULL TFX pipeline
echo.
python -m venv venv
call venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-tfx.txt
goto DONE

:MINIMAL
echo ⚠️ Python 3.11+ detected - Installing MINIMAL setup
echo    (Full TFX not available for Python 3.11+)
echo.
python -m venv venv
call venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-minimal.txt
goto DONE

:DONE
echo.
echo ============================================================
echo Installation complete!
echo ============================================================
echo.
echo Next steps:
echo   1. Activate environment: venv\Scripts\activate
echo   2. Start Jupyter: jupyter notebook tfx_pipeline.ipynb
echo   3. Run notebook cells sequentially
echo.
echo See SETUP_GUIDE.md for detailed information
echo ============================================================
pause
