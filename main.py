"""AI Video Watermark Remover - application entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_project_root() -> Path:
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def run_gui() -> int:
    """Launch the PySide6 graphical interface."""
    from PySide6.QtWidgets import QApplication

    from gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("AI Video Watermark Remover")
    window = MainWindow()
    window.show()
    return app.exec()


def run_cli(args: argparse.Namespace) -> int:
    """Run headless processing from the command line."""
    import numpy as np
    import traceback

    from processing.config_loader import get_masks_dir, get_output_dir, load_config
    from processing.mask_utils import load_mask
    from processing.pipeline import PipelineConfig, ProcessingPipeline
    from processing.video_reader import VideoReader

    config = load_config()
    proc_cfg = config.get("processing", {})

    backend = args.backend or proc_cfg.get("default_backend", "e2fgvi")
    chunk_size = args.chunk_size if args.chunk_size is not None else proc_cfg.get("default_chunk_size", "auto")
    if chunk_size != "auto":
        try:
            chunk_size = int(chunk_size)
        except ValueError:
            print(
                f"Invalid chunk size '{chunk_size}'. Use 'auto' or an integer.",
                file=sys.stderr,
            )
            return 1

    overlap = args.overlap if args.overlap is not None else int(proc_cfg.get("chunk_overlap", 5))
    crf = args.crf if args.crf is not None else int(proc_cfg.get("output_crf", 18))
    preserve_audio = not args.no_audio
    reencode = not args.no_reencode

    if args.verbose:
        print(f"Using backend={backend}, chunk_size={chunk_size}, overlap={overlap}, skip_seconds={args.skip_seconds}, crf={crf}")

    input_path = args.input
    mask_path = Path(args.mask)
    if args.mask_dir:
        mask_path = Path(args.mask_dir) / mask_path
    if not mask_path.is_absolute():
        mask_path = Path.cwd() / mask_path

    output_path = Path(args.output)
    if args.output_dir:
        output_path = Path(args.output_dir) / output_path
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    skip_seconds = None
    if getattr(args, "skip_seconds", None) is not None:
        skip_seconds = float(args.skip_seconds)
    elif getattr(args, "skip_frames", None) is not None:
        try:
            with VideoReader(input_path) as vr:
                fps = vr.metadata.fps or 30.0
            skip_seconds = float(args.skip_frames) / float(fps)
        except Exception as exc:
            print(f"Failed to determine skip frames: {exc}", file=sys.stderr)
            if args.verbose:
                traceback.print_exc()
            return 1
    else:
        skip_seconds = float(proc_cfg.get("skip_start_seconds", 0))

    try:
        mask = load_mask(mask_path)
    except Exception as exc:
        print(f"Error loading mask '{mask_path}': {exc}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1

    def on_progress(progress) -> None:
        print(
            f"\r{progress.message} [{progress.percent:.1f}%]",
            end="",
            flush=True,
        )

    try:
        pipeline = ProcessingPipeline(
            config=PipelineConfig(
                backend_name=backend,
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                preserve_audio=preserve_audio,
                reencode=reencode,
                output_crf=crf,
                skip_start_seconds=skip_seconds,
            ),
            progress_callback=on_progress,
        )
        result = pipeline.process(input_path, mask, output_path)
    except Exception as exc:
        print(f"Processing failed: {exc}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1

    print()
    if result.error:
        print(f"Error: {result.error}", file=sys.stderr)
        return 1
    if result.cancelled:
        print("Cancelled.")
        return 2

    print(result.message)
    return 0


def run_phase1_test(args: argparse.Namespace) -> int:
    """Verify chunk processing with passthrough backend."""
    import numpy as np

    from backends import get_backend
    from processing.chunk_processor import ChunkProcessor
    from processing.mask_utils import create_empty_mask
    from processing.video_reader import VideoReader

    reader = VideoReader(args.input)
    meta = reader.metadata
    mask = create_empty_mask(meta.width, meta.height)
    mask[meta.height // 4 : 3 * meta.height // 4, meta.width // 4 : 3 * meta.width // 4] = 255

    backend = get_backend("passthrough")
    backend.initialize()

    processor = ChunkProcessor(
        backend=backend,
        chunk_size=args.chunk_size,
        progress_callback=lambda p: print(p.message),
    )

    result = processor.process(args.input, args.output, mask)
    reader.close()

    if result.error:
        print(f"Phase 1 test failed: {result.error}", file=sys.stderr)
        return 1

    print(f"Phase 1 test passed: {args.output}")
    print(
        f"Processed {result.frames_processed} frames in "
        f"{result.current_chunk} chunks at {result.fps:.1f} FPS"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="AI Video Watermark Remover",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("gui", help="Launch graphical interface")

    process_parser = subparsers.add_parser("process", help="Process video from CLI")
    process_parser.add_argument("input", type=Path, help="Input video path")
    process_parser.add_argument("mask", type=Path, help="Mask PNG path")
    process_parser.add_argument("-o", "--output", type=Path, required=True)
    process_parser.add_argument(
        "-b",
        "--backend",
        choices=["e2fgvi", "propainter", "passthrough"],
        default=None,
    )
    process_parser.add_argument(
        "--chunk-size",
        default=None,
        help="Chunk size or 'auto'",
    )
    process_parser.add_argument("--overlap", type=int, default=None)
    process_parser.add_argument("--skip-seconds", type=float, default=None,
                                help="Skip this many seconds at start (overrides frames)")
    process_parser.add_argument("--skip-frames", type=int, default=None,
                                help="Skip this many frames at start")
    process_parser.add_argument("--crf", type=int, default=None)
    process_parser.add_argument("--output-dir", type=Path, default=None,
                                help="Optional base output directory for relative paths")
    process_parser.add_argument("--mask-dir", type=Path, default=None,
                                help="Optional base mask directory for relative paths")
    process_parser.add_argument("--no-audio", action="store_true")
    process_parser.add_argument("--no-reencode", action="store_true")
    process_parser.add_argument("--verbose", action="store_true")

    test_parser = subparsers.add_parser(
        "test-phase1",
        help="Verify chunk pipeline with passthrough backend",
    )
    test_parser.add_argument("input", type=Path)
    test_parser.add_argument("output", type=Path)
    test_parser.add_argument("--chunk-size", type=int, default=30)

    return parser


def main() -> int:
    """Application main entry."""
    _ensure_project_root()
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None or args.command == "gui":
        return run_gui()
    if args.command == "process":
        return run_cli(args)
    if args.command == "test-phase1":
        return run_phase1_test(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
