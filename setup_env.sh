#!/bin/bash
# Setup script for TFX Recommender Project (Linux/Mac)

echo "============================================================"
echo "TFX MovieLens Recommender - Environment Setup"
echo "============================================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.9+ from https://www.python.org/"
    exit 1
fi

echo ""
echo "Checking Python version..."
python3 --version

echo ""
echo "[1/5] Creating virtual environment..."
python3 -m venv venv

echo ""
echo "[2/5] Activating virtual environment..."
source venv/bin/activate

echo ""
echo "[3/5] Upgrading pip..."
pip install --upgrade pip

echo ""
echo "[4/5] Detecting Python version and installing dependencies..."
echo ""

# Try to install full TFX first (works for Python 3.9-3.10)
if pip install -r requirements-tfx.txt &> /dev/null; then
    echo ""
    echo "============================================================"
    echo "FULL TFX SETUP INSTALLED"
    echo "============================================================"
    echo "All TFX components available"
    echo "============================================================"
else
    echo "Full TFX installation failed. Installing minimal dependencies..."
    echo "(Your Python version might be 3.11+ which doesn't support full TFX)"
    echo ""
    pip install -r requirements-minimal.txt
    echo ""
    echo "============================================================"
    echo "MINIMAL SETUP INSTALLED"
    echo "============================================================"
    echo "TensorFlow + Core libraries installed"
    echo "Full TFX pipeline NOT available"
    echo "See SETUP_GUIDE.md for more information"
    echo "============================================================"
fi

echo ""
echo "[5/5] Verifying installation..."
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')" 2>/dev/null
python -c "import tfx; print(f'TFX: {tfx.__version__}')" 2>/dev/null || echo "TFX: Not installed (minimal setup)"

echo ""
echo "============================================================"
echo "Environment setup complete!"
echo "============================================================"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate:"
echo "  deactivate"
echo ""
echo "To start Jupyter Notebook:"
echo "  jupyter notebook tfx_pipeline.ipynb"
echo ""
echo "For detailed setup info, see: SETUP_GUIDE.md"
echo "============================================================"
