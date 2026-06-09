import os
import subprocess
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

def select_path(title, is_folder=True):
    """Opens a native dialog box to select files/folders easily."""
    root = tk.Tk()
    root.withdraw() # Hide the main tiny window
    root.attributes('-topmost', True)
    
    if is_folder:
        path = filedialog.askdirectory(title=title)
    else:
        path = filedialog.askopenfilename(
            title=title, 
            filetypes=[("PNG Images", "*.png"), ("All Files", "*.*")]
        )
    return Path(path) if path else None

def main():
    print("=== Video Watermark Remover - Batch Automation ===")
    
    # 1. Ask user for paths visually via native prompt windows
    input_dir = select_path("Select INPUT Folder Containing Videos", is_folder=True)
    if not input_dir: return print("Cancelled.")
    
    output_dir = select_path("Select OUTPUT Folder for Clean Videos", is_folder=True)
    if not output_dir: return print("Cancelled.")
    
    mask_file = select_path("Select the Mask PNG File", is_folder=False)
    if not mask_file: return print("Cancelled.")

    # 2. Grab all standard video assets in the target folder
    video_extensions = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
    videos = [f for f in input_dir.iterdir() if f.suffix.lower() in video_extensions]
    
    if not videos:
        print(f"\n❌ No matching videos found inside: {input_dir}")
        return

    print(f"\n🚀 Found {len(videos)} videos to process.")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Iterate and trigger main.py sequentially with your specific parameters
    for idx, video_path in enumerate(videos, start=1):
        out_video_path = output_dir / video_path.name
        
        print(f"\n--------------------------------------------------")
        print(f"[{idx}/{len(videos)}] Processing: {video_path.name}")
        print(f"Saving to: {out_video_path}")
        print(f"--------------------------------------------------")
        
        # Build your exact requested command arguments array
        cmd = [
            "python", "main.py", "process",
            str(video_path),
            str(mask_file),
            "-o", str(out_video_path),
            "--backend", "e2fgvi",
            "--crf", "18",
            "--chunk-size", "8",
            "--skip-frames", "0",
        ]
        
        # Run the command and pipe terminal updates straight to the console screen
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Error processing {video_path.name}. Moving to next file...")
            continue

    print("\n🎉 All batch items finished successfully!")

if __name__ == "__main__":
    main()