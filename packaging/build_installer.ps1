# Build Windows executable with PyInstaller
param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

Set-Location $Root

& $Python -m pip install pyinstaller --quiet

& $Python -m PyInstaller `
    --name "VideoWatermarkRemover" `
    --windowed `
    --noconfirm `
    --add-data "config.json;." `
    --hidden-import "PySide6.QtCore" `
    --hidden-import "PySide6.QtGui" `
    --hidden-import "PySide6.QtWidgets" `
    --collect-all "imageio_ffmpeg" `
    main.py

Write-Host "Build complete: dist\VideoWatermarkRemover\VideoWatermarkRemover.exe"
