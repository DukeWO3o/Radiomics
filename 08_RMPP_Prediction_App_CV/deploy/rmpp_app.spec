# -*- mode: python ; coding: utf-8 -*-
# =============================================================================
# PyInstaller spec - RMPP classification support system (CV pipeline build)
#
# Build inside 08_RMPP_Prediction_App_CV with conda env rmpp-pack:
#   pyinstaller deploy/rmpp_app.spec --noconfirm
# Output: dist/RMPP_Prediction/RMPP_Prediction.exe
# =============================================================================
import os
import sys

from PyInstaller.utils.hooks import collect_all

PROJECT_ROOT = os.path.dirname(os.path.abspath(SPECPATH))

datas = []
binaries = []
hiddenimports = []

# --- Streamlit stack ---
for pkg in ("streamlit", "altair", "plotly", "pyarrow", "pandas"):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h

# --- Medical imaging / radiomics C extensions ---
# Import names, not distribution names (pyradiomics -> radiomics).
for pkg in ("radiomics", "SimpleITK", "pydicom", "cv2", "pykwalify", "pywt"):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h

# --- Machine learning ---
for pkg in ("sklearn", "xgboost", "lightgbm", "joblib"):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h

# --- PyTorch (CPU build, largest component) ---
d, b, h = collect_all("torch")
datas += d; binaries += b; hiddenimports += h

# --- Application code and assets ---
datas += [
    (os.path.join(PROJECT_ROOT, "app.py"), "."),
    (os.path.join(PROJECT_ROOT, "unet_model.py"), "."),
    (os.path.join(PROJECT_ROOT, "config"), "config"),
    (os.path.join(PROJECT_ROOT, "models"), "models"),
    (os.path.join(PROJECT_ROOT, "demo"), "demo"),
]

# --- conda xgboost DLL lives outside site-packages ---
binaries += [
    (
        os.path.join(sys.prefix, "Library", "mingw-w64", "bin", "xgboost.dll"),
        os.path.join("Library", "mingw-w64", "bin"),
    ),
]

a = Analysis(
    [os.path.join(PROJECT_ROOT, "deploy", "run_app.py")],
    pathex=[PROJECT_ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "torchvision", "torchaudio", "PIL.ImageShow"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="RMPP_Prediction",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # Set RMPP_APP_CONSOLE=1 to build a console variant for troubleshooting.
    console=os.environ.get("RMPP_APP_CONSOLE", "0") == "1",
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="RMPP_Prediction",
)
