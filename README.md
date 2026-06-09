# AI Video Watermark Remover with Chunking

Windows desktop application for removing video watermarks using AI video inpainting using lower resources. Supports E2FGVI - Future support for ProPainter- and generic backends with chunked processing for long videos.

## Requirements

- Windows 11
- Python 3.10+
- NVIDIA GPU with CUDA
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

## Why Chunking?
Long-form video inpainting is notoriously difficult because AI models have strict limits on how many frames they can process at once. Most tools crash when processing anything longer than a few seconds. 

Our application uses **Intelligent Chunking** to solve this:

* **RAM Efficiency:** Instead of loading thousands of frams into RAM at once, the app processes the video in manageable, bite-sized segments resulting in 1 -3GB instead of dozens. This allows you to process high-resolution, long-form content even on consumer GPUs with limited hardware (like a GTX 1080).
* **High-Fidelity Quality:** By processing in smaller chunks, the model can maintain sharper focus on the masked area. We utilize a sliding window approach with frame-overlap, ensuring that the transition between one chunk and the next is seamless and flicker-free.
* **Stability:** If your computer encounters a hiccup, you aren't forced to restart a 2-hour render. The app manages the sequence, ensuring that each part of your video is treated with the same level of precision, regardless of the total video length.

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
