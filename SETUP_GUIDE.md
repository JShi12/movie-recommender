# Environment Setup Guide

## Step 1: Check Your Python Version

```bash
python --version
```

**Expected output:** `Python 3.x.x`

---

## Step 2: Choose Installation Method Based on Python Version

### 🟢 Option A: Python 3.9 or 3.10 (Recommended - Full TFX Support)

**Full TFX pipeline with all components:**

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements-tfx.txt
```

**You can run:** Complete TFX pipeline with all 8 components ✅

---

### 🟡 Option B: Python 3.11 or 3.12+ (Minimal Setup)

**TensorFlow + manual pipeline approach:**

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements-minimal.txt
```

**You can run:** TensorFlow model training, manual pipeline orchestration ⚠️

**Note:** Full TFX orchestration not available. You'll need to:
- Run pipeline components manually
- Use custom training scripts instead of TFX Trainer
- Implement custom evaluation logic

---

### 🔴 Option C: Install Python 3.10 (Best Experience)

If you want the full TFX experience:

1. **Download Python 3.10**: https://www.python.org/downloads/release/python-31011/
2. **Install** to a custom location (e.g., `C:\Python310`)
3. **Use specific Python version:**
   ```bash
   C:\Python310\python.exe -m venv venv
   venv\Scripts\activate
   pip install -r requirements-tfx.txt
   ```

---

## Step 3: Verify Installation

### For Full TFX Setup (Python 3.9-3.10):

```python
python -c "import tfx; print(f'TFX version: {tfx.__version__}')"
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
```

**Expected output:**
```
TFX version: 1.14.0
TensorFlow version: 2.13.0
```

### For Minimal Setup (Python 3.11+):

```python
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
python -c "import pandas as pd; print(f'Pandas version: {pd.__version__}')"
```

**Expected output:**
```
TensorFlow version: 2.15.x or higher
Pandas version: 1.5.x or higher
```

---

## Step 4: Launch Jupyter Notebook

```bash
jupyter notebook tfx_pipeline.ipynb
```

---

## Troubleshooting

### Error: "No matching distribution found for tfx"

**Solution:** You have Python 3.11+, which doesn't support TFX yet. Use Option B (minimal setup).

### Error: Import errors for TFX components

**Check:**
1. Python version: `python --version`
2. TFX installed: `pip list | grep tfx` (or `pip list | findstr tfx` on Windows)
3. Try reinstalling: `pip install --upgrade --force-reinstall tfx==1.14.0`

### Error: CUDA/GPU issues

**Solution:** Install CPU-only TensorFlow:
```bash
pip uninstall tensorflow
pip install tensorflow-cpu==2.13.0  # For Python 3.9-3.10
# or
pip install tensorflow-cpu>=2.15.0  # For Python 3.11+
```

### Error: "ModuleNotFoundError: No module named 'apache_beam'"

**Solution:**
```bash
pip install apache-beam[gcp]==2.44.0  # Python 3.9-3.10
# or
pip install apache-beam[gcp]>=2.50.0  # Python 3.11+
```

---

## Quick Reference

| Python Version | Requirements File | TFX Support | TensorFlow |
|----------------|-------------------|-------------|------------|
| 3.9 - 3.10 | requirements-tfx.txt | ✅ Full | 2.13.0 |
| 3.11 - 3.12+ | requirements-minimal.txt | ❌ No | 2.15.0+ |

---

## What's Next?

After successful installation:

1. **Open notebook:** `jupyter notebook tfx_pipeline.ipynb`
2. **Run cells sequentially** starting from cell 1
3. **Follow the plan** documented in the notebook

---

## Need Help?

- Check Python version compatibility table above
- Review error messages carefully
- Try minimal setup if full TFX doesn't work
- Consider installing Python 3.10 for full features
