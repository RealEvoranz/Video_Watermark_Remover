# AI Video Watermark Remover

Windows desktop application for removing video watermarks using AI video inpainting. Supports E2FGVI and ProPainter backends with chunked processing for long videos.

## Requirements

- Windows 11 (Linux-compatible design)
- Python 3.10+
- NVIDIA GPU with CUDA (recommended: 6GB+ VRAM)
- FFmpeg (or `imageio-ffmpeg` bundled via pip)

## Installation

```bash
cd video_watermark_remover
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Important:** The default `pip install torch` often installs a **CPU-only** build. Your RTX GPU will not be used until you install CUDA PyTorch:

**OLDER GPUS:** You may have to use https://download.pytorch.org/whl/cu126 instead of 128

```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Verify GPU detection:

```bash
python scripts/check_gpu.py
```

Verify all dependencies (app + E2FGVI + ProPainter):

```bash
python scripts/check_dependencies.py
```

You should see `cuda_available: True` and your GPU name (e.g. RTX 4050).

**Note:** `mmcv-full` cannot be installed on modern Python/Windows. The app includes a built-in mmcv compatibility shim for E2FGVI — you do **not** need to install mmcv manually.

## Usage

### GUI

```bash
python main.py gui
```

### CLI

```bash
python main.py process input.mp4 mask.png -o output.mp4 --backend e2fgvi
```

### Phase 1 verification (passthrough, no GPU models)

```bash
python main.py test-phase1 input.mp4 output.mp4 --chunk-size 30
```

## Workflow

1. Open a video (MP4, MOV, MKV, AVI)
2. Draw a mask with Rectangle, Brush, or Eraser tools
3. Save/load mask as grayscale PNG (255 = remove, 0 = preserve)
4. Select backend and chunk size (Auto adapts to VRAM)
5. Start processing — audio is preserved automatically

## Project Structure

```
video_watermark_remover/
├── main.py
├── run_batch.py   # Batch processing
├── config.json
├── gui/           # PySide6 interface
├── processing/    # Video I/O, chunking, FFmpeg, pipeline
├── backends/      # E2FGVI, ProPainter, passthrough
├── models/        # Download registry and resumable downloader
├── cache/         # Temporary processing files
└── output/        # Default output directory
```

## Model Downloads

Models are downloaded automatically on first use to `models/e2fgvi/` and `models/propainter/`.

| Backend | Source | Notes |
|---------|--------|-------|
| **ProPainter** | GitHub release v0.1.0 | Works out of the box |
| **E2FGVI** | Hugging Face mirror | No official GitHub release exists |

If E2FGVI auto-download fails, manually download `E2FGVI-HQ-CVPR22.pth` from [Google Drive](https://drive.google.com/file/d/10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3/view) and place it in `models/e2fgvi/weights/`.

## Packaging

For distribution, use PyInstaller:

```bash
pip install pyinstaller
pyinstaller --name "VideoWatermarkRemover" --windowed main.py
```
