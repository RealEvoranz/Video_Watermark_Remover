@echo off
title AI Video Watermark Remover Installer
color 0A

echo ==========================================
echo AI Video Watermark Remover Installer
echo ==========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.10+ is not installed or not in PATH.
    pause
    exit /b 1
)

echo Creating virtual environment...
python -m venv .venv

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo.
echo Upgrading pip...
python -m pip install --upgrade pip setuptools wheel

echo.
echo ==========================================
echo Installing CUDA 12.8 PyTorch
echo ==========================================
echo.

pip uninstall torch torchvision torchaudio -y >nul 2>&1

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

if errorlevel 1 (
    echo.
    echo CUDA 12.8 install failed.
    echo.
    echo If you have an older GPU try:
    echo pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
    echo.
    pause
    exit /b 1
)

echo.
echo ==========================================
echo Installing Application Requirements
echo ==========================================
echo.

pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo Failed to install requirements.
    pause
    exit /b 1
)

echo.
echo ==========================================
echo Verifying GPU Detection
echo ==========================================
echo.

if exist scripts\check_gpu.py (
    python scripts\check_gpu.py
)

echo.
echo ==========================================
echo Verifying Dependencies
echo ==========================================
echo.

if exist scripts\check_dependencies.py (
    python scripts\check_dependencies.py
)

echo.
echo ==========================================
echo Installation Complete
echo ==========================================
echo.
echo Launch GUI:
echo     .venv\Scripts\activate
echo     python main.py gui
echo.
pause