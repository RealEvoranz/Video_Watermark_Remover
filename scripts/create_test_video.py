"""Create a short test video for pipeline verification."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def create_test_video(output: Path, width: int = 640, height: int = 360, frames: int = 90) -> None:
    """Generate a synthetic MP4 with a visible watermark region."""
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30.0,
        (width, height),
    )

    for i in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (30 + (i % 50), 60, 90)
        cv2.putText(
            frame,
            f"Frame {i}",
            (50, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            "WATERMARK",
            (width - 200, height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        writer.write(frame)

    writer.release()
    print(f"Created test video: {output} ({frames} frames)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=Path, default=Path("cache/test_input.mp4"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--frames", type=int, default=90)
    args = parser.parse_args()
    create_test_video(args.output, args.width, args.height, args.frames)
