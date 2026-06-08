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

    from processing.mask_utils import load_mask
    from processing.pipeline import PipelineConfig, ProcessingPipeline

    mask = load_mask(args.mask)
    config = PipelineConfig(
        backend_name=args.backend,
        chunk_size="auto" if args.chunk_size == "auto" else int(args.chunk_size),
        chunk_overlap=args.overlap,
        preserve_audio=not args.no_audio,
        reencode=not args.no_reencode,
        output_crf=args.crf,
    )

    def on_progress(progress) -> None:
        print(
            f"\r{progress.message} [{progress.percent:.1f}%]",
            end="",
            flush=True,
        )

    pipeline = ProcessingPipeline(config=config, progress_callback=on_progress)
    result = pipeline.process(args.input, mask, args.output)
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
        default="e2fgvi",
    )
    process_parser.add_argument(
        "--chunk-size",
        default="auto",
        help="Chunk size or 'auto'",
    )
    process_parser.add_argument("--overlap", type=int, default=5)
    process_parser.add_argument("--crf", type=int, default=18)
    process_parser.add_argument("--no-audio", action="store_true")
    process_parser.add_argument("--no-reencode", action="store_true")

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
