@echo off
REM Setup script for TFX Recommender Project (Windows)

echo ============================================================
echo TFX MovieLens Recommender - Environment Setup
echo ============================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://www.python.org/
    exit /b 1
)

echo.
echo Checking Python version...
python --version

echo.
echo [1/5] Creating virtual environment...
python -m venv venv

echo.
echo [2/5] Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo [3/5] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [4/5] Detecting Python version and installing dependencies...
echo.

REM Try to install full TFX first (works for Python 3.9-3.10)
pip install -r requirements-tfx.txt >nul 2>&1

if errorlevel 1 (
    echo Full TFX installation failed. Installing minimal dependencies...
    echo ^(Your Python version might be 3.11+ which doesn't support full TFX^)
    echo.
    pip install -r requirements-minimal.txt
    echo.
    echo ============================================================
    echo MINIMAL SETUP INSTALLED
    echo ============================================================
    echo TensorFlow + Core libraries installed
    echo Full TFX pipeline NOT available
    echo See SETUP_GUIDE.md for more information
    echo ============================================================
) else (
    echo.
    echo ============================================================
    echo FULL TFX SETUP INSTALLED
    echo ============================================================
    echo All TFX components available
    echo ============================================================
)

echo.
echo [5/5] Verifying installation...
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')" 2>nul
python -c "import tfx; print(f'TFX: {tfx.__version__}')" 2>nul || echo TFX: Not installed (minimal setup)

echo.
echo ============================================================
echo Environment setup complete!
echo ============================================================
echo.
echo To activate the environment:
echo   venv\Scripts\activate.bat
echo.
echo To deactivate:
echo   deactivate
echo.
echo To start Jupyter Notebook:
echo   jupyter notebook tfx_pipeline.ipynb
echo.
echo For detailed setup info, see: SETUP_GUIDE.md
echo ============================================================
