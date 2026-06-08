"""ProPainter video inpainting backend."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import scipy.ndimage
import torch
from PIL import Image

from backends.base_backend import BaseBackend, ProgressCallback
from processing.config_loader import get_models_dir
from processing.gpu_utils import require_cuda


class ProPainterBackend(BaseBackend):
    """Propagation and Transformer video inpainting."""

    name = "propainter"
    display_name = "ProPainter"
    description = "High-quality propagation-based inpainting (ICCV 2023)"
    requires_gpu = True
    min_vram_mb = 5120

    def __init__(self) -> None:
        super().__init__()
        self._device: torch.device | None = None
        self._model = None
        self._fix_raft = None
        self._fix_flow_complete = None
        self._to_tensors = None
        self._source_dir: Path | None = None
        self._weights_dir: Path | None = None
        self._use_fp16 = True
        self._neighbor_length = 8
        self._mask_dilates = 5
        self._flow_mask_dilates = 8

    def initialize(self, progress_callback: ProgressCallback | None = None) -> None:
        if progress_callback:
            progress_callback("Downloading ProPainter models...")

        from models.downloader import ModelDownloader

        downloader = ModelDownloader(get_models_dir())
        entry = downloader.ensure_backend("propainter")

        self._source_dir = entry.source_dir(get_models_dir())
        self._weights_dir = entry.weights_dir(get_models_dir())

        if str(self._source_dir) not in sys.path:
            sys.path.insert(0, str(self._source_dir))

        if progress_callback:
            progress_callback("Loading ProPainter models...")

        from core.utils import to_tensors
        from model.modules.flow_comp_raft import RAFT_bi
        from model.propainter import InpaintGenerator
        from model.recurrent_flow_completion import RecurrentFlowCompleteNet

        self._to_tensors = to_tensors
        require_cuda("ProPainter")
        self._device = torch.device("cuda")

        self._fix_raft = RAFT_bi(
            str(self._weights_dir / "raft-things.pth"),
            self._device,
        )
        self._fix_raft.eval()

        self._fix_flow_complete = RecurrentFlowCompleteNet(
            str(self._weights_dir / "recurrent_flow_completion.pth")
        )
        self._fix_flow_complete.to(self._device)
        self._fix_flow_complete.eval()

        self._model = InpaintGenerator(
            model_path=str(self._weights_dir / "ProPainter.pth")
        ).to(self._device)
        self._model.eval()

        self._initialized = True
        if progress_callback:
            progress_callback("ProPainter ready")

    def _prepare_masks(
        self,
        mask: np.ndarray,
        size: tuple[int, int],
        length: int,
    ) -> tuple[list[Image.Image], list[Image.Image]]:
        width, height = size
        if mask.shape[1] != width or mask.shape[0] != height:
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

        mask_img = np.array(mask.convert("L") if isinstance(mask, Image.Image) else mask)

        if self._flow_mask_dilates > 0:
            flow_mask = scipy.ndimage.binary_dilation(
                mask_img, iterations=self._flow_mask_dilates
            ).astype(np.uint8)
        else:
            flow_mask = (mask_img > 0).astype(np.uint8)

        if self._mask_dilates > 0:
            dilated = scipy.ndimage.binary_dilation(
                mask_img, iterations=self._mask_dilates
            ).astype(np.uint8)
        else:
            dilated = (mask_img > 0).astype(np.uint8)

        flow_masks = [Image.fromarray(flow_mask * 255)] * length
        masks_dilated = [Image.fromarray(dilated * 255)] * length
        return flow_masks, masks_dilated

    def _resize_frames(
        self,
        frames: list[Image.Image],
    ) -> tuple[list[Image.Image], tuple[int, int], tuple[int, int]]:
        out_size = frames[0].size
        process_size = (
            out_size[0] - out_size[0] % 8,
            out_size[1] - out_size[1] % 8,
        )
        if out_size != process_size:
            frames = [f.resize(process_size) for f in frames]
        return frames, process_size, out_size

    def process_chunk(
        self,
        frames: list[np.ndarray],
        mask: np.ndarray,
    ) -> list[np.ndarray]:
        self.ensure_initialized()
        if self.is_cancelled or not frames:
            return [f.copy() for f in frames]

        assert self._model is not None
        assert self._fix_raft is not None
        assert self._fix_flow_complete is not None
        assert self._to_tensors is not None
        assert self._device is not None

        pil_frames = [
            Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames
        ]
        pil_frames, process_size, out_size = self._resize_frames(pil_frames)
        video_length = len(pil_frames)

        flow_masks, masks_dilated = self._prepare_masks(
            mask, process_size, video_length
        )

        frames_t = (
            self._to_tensors()(pil_frames).unsqueeze(0).to(self._device) * 2 - 1
        )
        flow_masks_t = self._to_tensors()(flow_masks).unsqueeze(0).to(self._device)
        masks_t = self._to_tensors()(masks_dilated).unsqueeze(0).to(self._device)

        if self._use_fp16:
            frames_t = frames_t.half()
            flow_masks_t = flow_masks_t.half()
            masks_t = masks_t.half()

        comp_frames = [np.array(f).astype(np.uint8) for f in pil_frames]

        with torch.no_grad():
            if frames_t.size(-1) <= 640:
                short_clip_len = 12
            elif frames_t.size(-1) <= 720:
                short_clip_len = 8
            elif frames_t.size(-1) <= 1280:
                short_clip_len = 4
            else:
                short_clip_len = 2

            if video_length > short_clip_len:
                gt_flows_f_list, gt_flows_b_list = [], []
                for f in range(0, video_length, short_clip_len):
                    if self.is_cancelled:
                        return [frame.copy() for frame in frames]
                    end_f = min(video_length, f + short_clip_len)
                    if f == 0:
                        flows_f, flows_b = self._fix_raft(
                            frames_t[:, f:end_f], iters=20
                        )
                    else:
                        flows_f, flows_b = self._fix_raft(
                            frames_t[:, f - 1 : end_f], iters=20
                        )
                        flows_f = flows_f[:, 1:]
                        flows_b = flows_b[:, 1:]
                    gt_flows_f_list.append(flows_f)
                    gt_flows_b_list.append(flows_b)
                    torch.cuda.empty_cache()

                gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
                gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
                gt_flows_bi = (gt_flows_f, gt_flows_b)
            else:
                gt_flows_bi = self._fix_raft(frames_t, iters=20)
                torch.cuda.empty_cache()

            if self._use_fp16:
                frames_t, flow_masks_t, masks_t = (
                    frames_t.half(),
                    flow_masks_t.half(),
                    masks_t.half(),
                )
                gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())
                self._fix_flow_complete.half()
                self._model.half()
            else:
                self._fix_flow_complete.float()
                self._model.float()

            flow_length = gt_flows_bi[0].size(1)
            pred_flows_bi, _ = self._fix_flow_complete.forward_bidirect_flow(
                (flow_masks_t[:, :flow_length], gt_flows_bi)
            )
            pred_flows_bi = self._fix_flow_complete.combine_flow(
                (flow_masks_t[:, :flow_length], pred_flows_bi, gt_flows_bi)
            )

            masked_frames = frames_t * (1 - masks_t)
            neighbor_stride = self._neighbor_length // 2

            for f_idx in range(0, video_length, neighbor_stride):
                if self.is_cancelled:
                    break

                neighbor_ids = list(
                    range(
                        max(0, f_idx - neighbor_stride),
                        min(video_length, f_idx + neighbor_stride + 1),
                    )
                )
                selected_imgs = masked_frames[:, neighbor_ids, :, :, :]
                selected_masks = masks_t[:, neighbor_ids, :, :, :]
                selected_update_masks = flow_masks_t[:, neighbor_ids, :, :, :]
                selected_pred_flows_bi = (
                    pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :],
                    pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :],
                )

                l_t = len(neighbor_ids)
                pred_img = self._model(
                    selected_imgs,
                    selected_pred_flows_bi,
                    selected_masks,
                    selected_update_masks,
                    l_t,
                )

                pred_img = pred_img[:, :, : process_size[1], : process_size[0]]
                pred_img = (pred_img + 1) / 2
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255

                for i, idx in enumerate(neighbor_ids):
                    binary = np.expand_dims(
                        (np.array(masks_dilated[idx]) > 0).astype(np.uint8), 2
                    )
                    img = (
                        pred_img[i].astype(np.uint8) * binary
                        + comp_frames[idx] * (1 - binary)
                    )
                    if process_size != out_size:
                        img = np.array(
                            Image.fromarray(img).resize(out_size, Image.BILINEAR)
                        )
                    comp_frames[idx] = img

        result = [
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in comp_frames
        ]

        if self._device.type == "cuda":
            torch.cuda.empty_cache()

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
            base = 12 if vram >= 8192 else 8 if vram >= 6144 else 4
        else:
            base = 6 if vram >= 8192 else 4

        return max(4, base)

    def cleanup(self) -> None:
        self._model = None
        self._fix_raft = None
        self._fix_flow_complete = None
        if self._device and self._device.type == "cuda":
            torch.cuda.empty_cache()
        super().cleanup()
