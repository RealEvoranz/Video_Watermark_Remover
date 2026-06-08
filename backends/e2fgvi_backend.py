"""E2FGVI-HQ video inpainting backend optimized for low-VRAM deployment."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from backends.base_backend import BaseBackend, ProgressCallback
from backends.mmcv_compat import ensure_mmcv_compat
from processing.config_loader import get_models_dir
from processing.gpu_utils import require_cuda


class E2FGVIBackend(BaseBackend):
    """Flow-guided video inpainting using E2FGVI-HQ with low-VRAM optimizations."""

    name = "e2fgvi"
    display_name = "E2FGVI-HQ"
    description = "Fast flow-guided video inpainting (CVPR 2022) - Low VRAM Mode"
    requires_gpu = True
    min_vram_mb = 4096

    def __init__(self) -> None:
        super().__init__()
        self._device: torch.device | None = None
        self._model = None
        self._to_tensors = None
        self._source_dir: Path | None = None
        self._ckpt_path: Path | None = None
        self._neighbor_stride = 20
        self._ref_length = 10

    def initialize(self, progress_callback: ProgressCallback | None = None) -> None:
        if progress_callback:
            progress_callback("Downloading E2FGVI models...")

        from models.downloader import ModelDownloader

        downloader = ModelDownloader(get_models_dir())
        entry = downloader.ensure_backend("e2fgvi")

        self._source_dir = entry.source_dir(get_models_dir())
        self._ckpt_path = entry.weights_dir(get_models_dir()) / "E2FGVI-HQ-CVPR22.pth"

        ensure_mmcv_compat()

        if str(self._source_dir) not in sys.path:
            sys.path.insert(0, str(self._source_dir))

        if progress_callback:
            progress_callback("Loading E2FGVI model...")

        require_cuda("E2FGVI")
        self._device = torch.device("cuda")

        utils = importlib.import_module("core.utils")
        self._to_tensors = utils.to_tensors()

        model_module = importlib.import_module("model.e2fgvi_hq")
        self._model = model_module.InpaintGenerator().to(self._device)
        state = torch.load(self._ckpt_path, map_location=self._device, weights_only=False)
        self._model.load_state_dict(state)
        self._model.eval()

        self._initialized = True
        if progress_callback:
            progress_callback("E2FGVI ready")

    def _get_ref_index(
        self,
        frame_idx: int,
        neighbor_ids: list[int],
        length: int,
    ) -> list[int]:
        ref_index: list[int] = []
        for i in range(0, length, self._ref_length):
            if i not in neighbor_ids:
                ref_index.append(i)
        return ref_index

    def _prepare_mask(self, mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        """Prepare dilated binary mask for E2FGVI."""
        width, height = size
        if mask.shape[1] != width or mask.shape[0] != height:
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        binary = (mask > 0).astype(np.uint8)
        dilated = cv2.dilate(
            binary,
            cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
            iterations=4,
        )
        return dilated

    def process_chunk(
        self,
        frames: list[np.ndarray],
        mask: np.ndarray,
    ) -> list[np.ndarray]:
        self.ensure_initialized()
        if self.is_cancelled or not frames:
            return [f.copy() for f in frames]

        assert self._model is not None
        assert self._device is not None
        assert self._to_tensors is not None

        orig_height, orig_width = frames[0].shape[:2]
        
        # --- LOW-VRAM OPTIMIZATION: INTERNAL DOWNSCALING ---
        # Caps temporal attention mapping complexity to a maximum of 720p 
        # to guarantee execution boundaries within your 6GB VRAM target.
        target_width, target_height = orig_width, orig_height
        if orig_height > 720:
            target_height = 720
            target_width = int(orig_width * (720 / orig_height))
            target_width = (target_width // 4) * 4  # Enforce basic 4-pixel structural network alignment

        # Convert native high-res frames smoothly to target calculation resolution profiles
        pil_frames = [
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize(
                (target_width, target_height), Image.Resampling.BILINEAR
            )
            for frame in frames
        ]
        
        # Keep native high-res mask layers intact for sharp outer boundary alpha-blends
        binary_mask_native = (mask > 0).astype(np.uint8)
        binary_masks_native = [np.expand_dims(binary_mask_native, 2) for _ in range(len(frames))]
        
        # Build downscaled tracking mask elements for local convolution iterations
        binary_mask_internal = self._prepare_mask(mask, (target_width, target_height))
        video_length = len(frames)
        rgb_frames_native = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]

        # Structure input arrays into standard 5D PyTorch spaces: [B, T, C, H, W]
        imgs = self._to_tensors(pil_frames).unsqueeze(0) * 2 - 1
        mask_pil = Image.fromarray(binary_mask_internal * 255)
        masks = self._to_tensors([mask_pil] * video_length).unsqueeze(0)

        imgs = imgs.to(self._device)
        masks = masks.to(self._device)

        comp_frames: list[np.ndarray | None] = [None] * video_length
        
        # Calculate standard block divisibility requirements
        mod_size_h, mod_size_w = 60, 108
        h_pad = (mod_size_h - target_height % mod_size_h) % mod_size_h
        w_pad = (mod_size_w - target_width % mod_size_w) % mod_size_w

        # Flush any hanging hardware cache layers before starting heavy forward iterations
        torch.cuda.empty_cache()

        with torch.no_grad():
            # --- LOW-VRAM OPTIMIZATION: MIXED-PRECISION RUNTIME ---
            # Drops inference tensor bit depth tracking requirements cleanly in half (FP32 -> FP16)
            with torch.cuda.amp.autocast(enabled=True):
                for f in range(0, video_length, self._neighbor_stride):
                    if self.is_cancelled:
                        break

                    neighbor_ids = list(
                        range(
                            max(0, f - self._neighbor_stride),
                            min(video_length, f + self._neighbor_stride + 1),
                        )
                    )
                    ref_ids = self._get_ref_index(f, neighbor_ids, video_length)
                    
                    selected_imgs = imgs[:, neighbor_ids + ref_ids, :, :, :]
                    selected_masks = masks[:, neighbor_ids + ref_ids, :, :, :]

                    masked_imgs = selected_imgs * (1 - selected_masks)
                    
                    # --- 5D TO 4D RESHAPE FOR REFLECTION PADDING COMPATIBILITY ---
                    # Safely handles PyTorch non-constant spatial padding limitations
                    if h_pad > 0 or w_pad > 0:
                        B_val, T_val, C_val, H_val, W_val = masked_imgs.shape
                        # Reshape down to 4D: [B*T, C, H, W]
                        masked_imgs = masked_imgs.view(B_val * T_val, C_val, H_val, W_val)
                        
                        # Apply spatial reflection padding to spatial dimensions safely
                        masked_imgs = torch.nn.functional.pad(
                            masked_imgs, (0, w_pad, 0, h_pad), mode='reflect'
                        )
                        
                        # Restore structural 5D layout expected by network transformer blocks: [B, T, C, H_padded, W_padded]
                        masked_imgs = masked_imgs.view(B_val, T_val, C_val, H_val + h_pad, W_val + w_pad)

                    # Execute forward inference loop pass
                    pred_imgs, _ = self._model(masked_imgs, len(neighbor_ids))
                    
                    # Crop back down to calculated operational target boundaries
                    pred_imgs = pred_imgs[:, :, :target_height, :target_width]
                    pred_imgs = (pred_imgs + 1) / 2
                    
                    # Shift processed arrays cleanly back to standard CPU workspace formats
                    pred_imgs = pred_imgs.permute(0, 2, 3, 1).cpu().float().numpy() * 255.0
                    pred_imgs = np.clip(pred_imgs, 0, 255).astype(np.uint8)

                    for i, idx in enumerate(neighbor_ids):
                        # Upscale structural patch output back to native video format matching original size profiles
                        pred_native_rgb = cv2.resize(
                            pred_imgs[i], (orig_width, orig_height), interpolation=cv2.INTER_CUBIC
                        )

                        # Composite the generated clean frame over the original background image
                        blended = (
                            pred_native_rgb * binary_masks_native[idx]
                            + rgb_frames_native[idx] * (1 - binary_masks_native[idx])
                        )
                        
                        if comp_frames[idx] is None:
                            comp_frames[idx] = blended
                        else:
                            comp_frames[idx] = (
                                comp_frames[idx].astype(np.float32) * 0.5
                                + blended.astype(np.float32) * 0.5
                            ).astype(np.uint8)

        # Deallocate high memory reference counts and free VRAM allocations immediately
        del imgs, masks
        torch.cuda.empty_cache()

        # Reconstruct final arrays back into default OpenCV native BGR configurations
        result: list[np.ndarray] = []
        for idx, frame in enumerate(comp_frames):
            if frame is None:
                result.append(frames[idx].copy())
            else:
                result.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        return result

    def estimate_chunk_size(
        self,
        width: int,
        height: int,
        available_vram_mb: int | None = None,
    ) -> int:
        pixels = width * height
        vram = available_vram_mb or 6144

        if pixels <= 1280 * 720:
            base = 20 if vram >= 8192 else 12 if vram >= 6144 else 8
        elif pixels <= 1920 * 1080:
            base = 14 if vram >= 8192 else 8 if vram >= 6144 else 5
        else:
            base = 6 if vram >= 8192 else 2

        return max(2, base)

    def cleanup(self) -> None:
        self._model = None
        if self._device and self._device.type == "cuda":
            torch.cuda.empty_cache()
        super().cleanup()