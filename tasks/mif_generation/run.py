#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HEMIT H&E-to-mIF image reconstruction using multi-stage SpaVista features and a U-Net-style decoder.


Core design:
  - The encoder is frozen.
  - SpaVista uses the MambaVision pyramid: patch_embed + levels + final norm,
    selecting one feature per spatial resolution, default N=4.
  - A U-Net/FPN-style decoder is trained on frozen SpaVista features:
      per-level 1x1 adapters -> top-down fusion with skip connections -> final conv head.
  - HEMIT H&E and mIF patches are resized to --image_size, default 224.
  - Default output channels are panCK and CD3. DAPI can be included by --out_channels panCK,CD3,DAPI.

The released metadata CSV contains:
  split, tile_id, he_path, label_path, ...

Outputs:
  results/<tag>/
    run_config.json
    all_model_test_metrics.csv
    all_model_channel_metrics.csv
    <MODEL>/
      best_decoder.pt
      last_decoder.pt
      train_log.csv
      val_metrics_overall.csv
      val_metrics_per_channel.csv
      test_metrics_overall.csv
      test_metrics_per_channel.csv
      test_examples/
        example_*.png

This module reuses SpaVista checkpoint loading utilities from tasks.protein_abundance.run.
"""

import argparse
import os
import warnings
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from tqdm import tqdm

from spavista.config import load_path_config

try:
    import matplotlib.pyplot as plt
except Exception as e:
    raise ImportError("This script needs matplotlib. Install it with: pip install matplotlib") from e

try:
    from tasks.protein_abundance.run import (
        str2bool,
        seed_torch,
        seed_worker,
        ensure_dir,
        check_file,
        add_project_root,
        save_run_config,
        get_device,
        autocast_context,
        build_encoder,
        sanitize_name,
        resolve_hemit_path,
    )
except Exception as e:
    raise ImportError(
        "Cannot import utilities from tasks.protein_abundance.run. "
        "Run this module from the repository root."
    ) from e

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
# =============================================================================
# Helpers
# =============================================================================
def parse_comma_list(x: str) -> List[str]:
    if x is None or str(x).strip() == "":
        return []
    return [v.strip() for v in str(x).split(",") if v.strip() != ""]


def channel_indices_from_names(channel_names: List[str]) -> Tuple[List[int], List[str]]:
    """HEMIT RGB mapping: R=panCK, G=CD3, B=DAPI."""
    mapping = {
        "panck": 0,
        "panCK": 0,
        "cd3": 1,
        "CD3": 1,
        "dapi": 2,
        "DAPI": 2,
    }
    indices = []
    canonical = []
    for name in channel_names:
        if name not in mapping:
            low = name.lower()
            if low not in mapping:
                raise ValueError(f"Unsupported channel name: {name}. Use panCK, CD3, DAPI.")
            idx = mapping[low]
        else:
            idx = mapping[name]
        indices.append(idx)
        if idx == 0:
            canonical.append("panCK")
        elif idx == 1:
            canonical.append("CD3")
        elif idx == 2:
            canonical.append("DAPI")
    return indices, canonical


def tensor_to_numpy_img(x: torch.Tensor) -> np.ndarray:
    """Convert [C,H,W] tensor in [0,1] to HWC uint8."""
    x = x.detach().float().cpu().clamp(0, 1)
    if x.shape[0] == 1:
        arr = x[0].numpy()
        arr = np.stack([arr, arr, arr], axis=-1)
    elif x.shape[0] == 2:
        arr = torch.zeros(3, x.shape[1], x.shape[2])
        arr[0] = x[0]  # panCK red
        arr[1] = x[1]  # CD3 green
        arr = arr.permute(1, 2, 0).numpy()
    else:
        arr = x[:3].permute(1, 2, 0).numpy()
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def denormalize_he(x: torch.Tensor) -> torch.Tensor:
    """ImageNet denormalization for H&E image tensor [3,H,W]."""
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device).view(3, 1, 1)
    return (x * std + mean).clamp(0, 1)




# =============================================================================
# Dataset
# =============================================================================
class HEMITImageReconstructionDataset(Dataset):
    """
    HEMIT paired H&E and mIF/mIHC image dataset.

    HEMIT label RGB mapping:
        R = panCK
        G = CD3
        B = DAPI
    """

    def __init__(
        self,
        csv_path: str,
        split: str,
        out_channels: List[str],
        data_root: str = "",
        image_size: int = 224,
        normal: bool = True,
        strict_path_check: bool = True,
        verbose: bool = True,
    ):
        super().__init__()
        check_file(csv_path, "HEMIT csv")
        self.csv_path = csv_path
        self.split = str(split)
        self.out_channels = list(out_channels)
        self.channel_indices, self.channel_names = channel_indices_from_names(self.out_channels)
        self.image_size = int(image_size)
        self.normal = bool(normal)
        self.verbose = bool(verbose)

        df = pd.read_csv(csv_path)
        for col in ["split", "he_path", "label_path"]:
            if col not in df.columns:
                raise ValueError(f"CSV must contain column: {col}")
        if "tile_id" not in df.columns:
            df["tile_id"] = np.arange(len(df)).astype(str)

        df = df[df["split"].astype(str) == self.split].copy()
        if len(df) == 0:
            raise RuntimeError(f"No samples found for split={split} in {csv_path}")

        df["he_path"] = df["he_path"].map(
            lambda path: resolve_hemit_path(path, data_root, self.split, "input")
        )
        df["label_path"] = df["label_path"].map(
            lambda path: resolve_hemit_path(path, data_root, self.split, "label")
        )

        if strict_path_check:
            he_exists = df["he_path"].map(lambda p: os.path.exists(str(p))).values
            label_exists = df["label_path"].map(lambda p: os.path.exists(str(p))).values
            keep = he_exists & label_exists
            n_drop = int((~keep).sum())
            if n_drop > 0:
                print(f"[WARN] split={split}: drop {n_drop} samples with missing paths.")
            df = df.loc[keep].copy()

        if len(df) == 0:
            raise RuntimeError(f"No valid samples left for split={split}.")

        self.df = df.reset_index(drop=True)

        if self.verbose:
            print("-" * 80)
            print(f"[DATASET] HEMIT reconstruction split={self.split}")
            print(f"[DATASET] samples={len(self.df):,}")
            print(f"[DATASET] image_size={self.image_size}, normal={self.normal}")
            print(f"[DATASET] output channels={self.channel_names}")
            print("-" * 80)

    def __len__(self):
        return len(self.df)

    def _read_rgb(self, path: str) -> Image.Image:
        img = Image.open(path).convert("RGB")
        img.load()
        return img

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        he = self._read_rgb(str(row["he_path"]))
        lab = self._read_rgb(str(row["label_path"]))

        if he.size != (self.image_size, self.image_size):
            he = he.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        if lab.size != (self.image_size, self.image_size):
            lab = lab.resize((self.image_size, self.image_size), resample=Image.BILINEAR)

        he_t = TF.to_tensor(he)
        if self.normal:
            he_t = TF.normalize(he_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        lab_np = np.array(lab).astype(np.float32) / 255.0
        target_np = lab_np[..., self.channel_indices]  # H,W,Cout
        target_t = torch.from_numpy(target_np).permute(2, 0, 1).float()

        return he_t, target_t, int(idx)


# =============================================================================
# Multi-stage feature extractor
# =============================================================================
class MultiStageFeatureExtractor(nn.Module):
    """
    Convert each frozen FM to a list of spatial feature maps ordered shallow -> deep.

    SpaVista:
        Manually traces patch_embed + MambaVision levels + final norm, then selects
        one feature per spatial resolution. This avoids the old 28x28/14x14/7x7/7x7
        issue and usually yields a proper 56x56/28x28/14x14/7x7 pyramid.

    ViT FMs:
        ViTs have no spatial-resolution stages. This script takes the last
        --num_feature_levels transformer block outputs and reshapes patch tokens to maps.
        These maps usually have the same HxW, e.g. 16x16 for 224/14.
    """

    KNOWN_CHANNELS = {
        32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512,
        640, 768, 896, 960, 1024, 1152, 1280, 1408, 1536, 2048, 3072, 4096,
    }

    def __init__(
        self,
        encoder: nn.Module,
        model_name: str,
        num_feature_levels: int = 4,
    ):
        super().__init__()
        self.encoder = encoder
        self.model_name = str(model_name)
        self.num_feature_levels = int(num_feature_levels)
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

    @staticmethod
    def _unwrap_model(encoder: nn.Module) -> nn.Module:
        return encoder.model if hasattr(encoder, "model") else encoder

    @classmethod
    def _to_bchw(cls, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise RuntimeError(f"Expected 4D feature map, got {tuple(x.shape)}")
        # If NHWC, convert to NCHW.
        if x.shape[1] not in cls.KNOWN_CHANNELS and x.shape[-1] in cls.KNOWN_CHANNELS:
            x = x.permute(0, 3, 1, 2).contiguous()
        return x.contiguous()

    def _spavista_multistage_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract a true pyramid from SpaVista/MambaVision.

        Important:
        A naive implementation used encoder._forward_all_levels(x) and then took maps[-N].
        In the MambaVision release used by SpaVista, this can produce:
            28x28 -> 14x14 -> 7x7 -> 7x7
        because the patch_embed 56x56 feature is not included and the last two
        level outputs may share the same 7x7 resolution.

        For dense mIF reconstruction, we instead manually trace:
            patch_embed output + each MambaVision level output + final norm output
        then keep one feature per spatial resolution, preferring the deepest feature
        if multiple maps have the same HxW. This usually yields:
            56x56 -> 28x28 -> 14x14 -> 7x7
        for the released SpaVista checkpoint.
        """
        # Prefer the SpaVistaEncoder helper because it unwraps MambaVisionForIBOT correctly.
        if hasattr(self.encoder, "_get_mambavision_model"):
            model = self.encoder._get_mambavision_model()
        else:
            model = self._unwrap_model(self.encoder)

        if not hasattr(model, "patch_embed") or not hasattr(model, "levels"):
            # Compatibility fallback for compatible encoder wrappers.
            if not hasattr(self.encoder, "_forward_all_levels"):
                raise RuntimeError(
                    "Cannot extract SpaVista pyramid: neither model.patch_embed/model.levels "
                    "nor encoder._forward_all_levels is available."
                )
            outs = self.encoder._forward_all_levels(x)
            maps = [self._to_bchw(o) for o in outs if torch.is_tensor(o) and o.dim() == 4]
        else:
            maps = []

            # 1) High-resolution stem/patch-embedding feature, typically [B,128,56,56].
            feat = model.patch_embed(x)
            if torch.is_tensor(feat) and feat.dim() == 4:
                maps.append(self._to_bchw(feat))

            # 2) Stage/level features.
            # Different MambaVision releases use different Level.forward return formats:
            #   - Tensor: already the feature passed to the next level.
            #   - Tuple/List: commonly (x_next, x_stage). We use x_next for continuing
            #     the forward pass and append the stage tensor when available.
            x_level = feat
            for level in model.levels:
                out = level(x_level)

                if isinstance(out, (tuple, list)):
                    tensor_outs = [o for o in out if torch.is_tensor(o) and o.dim() == 4]
                    if len(tensor_outs) == 0:
                        raise RuntimeError("MambaVision level returned tuple/list but no 4D tensor was found.")

                    # Continue with the first tensor, matching official-style (x_next, x_stage).
                    x_next = tensor_outs[0]

                    # For skip features, prefer the second tensor if present because it is often
                    # the current stage output before downsampling. Otherwise use the first.
                    stage_feat = tensor_outs[1] if len(tensor_outs) >= 2 else tensor_outs[0]
                    maps.append(self._to_bchw(stage_feat))
                    x_level = x_next
                elif torch.is_tensor(out):
                    x_level = out
                    if out.dim() == 4:
                        maps.append(self._to_bchw(out))
                else:
                    raise RuntimeError(f"Unsupported MambaVision level output type: {type(out)}")

            # 3) Final normalized feature. This is usually the best/deepest 7x7 map.
            if hasattr(model, "norm") and model.norm is not None and torch.is_tensor(x_level):
                try:
                    norm_feat = model.norm(x_level)
                    if torch.is_tensor(norm_feat) and norm_feat.dim() == 4:
                        maps.append(self._to_bchw(norm_feat))
                except Exception as e:
                    print(f"[WARN] SpaVista final norm could not be applied to multistage feature: {e}")

        if len(maps) == 0:
            raise RuntimeError("SpaVista multistage extraction produced no feature maps.")

        # Keep one feature per spatial resolution. If a resolution appears multiple times
        # such as 7x7 before and after the final level, keep the deepest/latest one.
        latest_by_hw = {}
        for feat in maps:
            feat = self._to_bchw(feat)
            hw = (int(feat.shape[-2]), int(feat.shape[-1]))
            latest_by_hw[hw] = feat

        selected = list(latest_by_hw.values())
        selected = sorted(selected, key=lambda t: int(t.shape[-2]) * int(t.shape[-1]), reverse=True)

        if len(selected) < self.num_feature_levels:
            raise RuntimeError(
                f"SpaVista has only {len(selected)} unique spatial levels "
                f"{[tuple(f.shape) for f in selected]}, but num_feature_levels={self.num_feature_levels}."
            )

        return selected[:self.num_feature_levels]




    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        self.encoder.eval()
        with torch.no_grad():
            maps = self._spavista_multistage_maps(x)
        return [feature.detach() for feature in maps]


# =============================================================================
# U-Net / FPN decoder
# =============================================================================
class NormActConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str = "gn"):
        super().__init__()
        if norm == "bn":
            norm1 = nn.BatchNorm2d(out_ch)
            norm2 = nn.BatchNorm2d(out_ch)
        elif norm == "gn":
            groups = min(8, out_ch)
            while out_ch % groups != 0 and groups > 1:
                groups -= 1
            norm1 = nn.GroupNorm(groups, out_ch)
            norm2 = nn.GroupNorm(groups, out_ch)
        elif norm == "none":
            norm1 = nn.Identity()
            norm2 = nn.Identity()
        else:
            raise ValueError("norm must be bn, gn, or none")

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            norm1,
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            norm2,
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


_RESIZE_WEIGHT_CACHE = {}


def _linear_resize_weights(input_size: int, output_size: int, device, dtype):
    key = (int(input_size), int(output_size), str(device), dtype)
    cached = _RESIZE_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached

    scale = float(input_size) / float(output_size)
    source = (torch.arange(output_size, dtype=torch.float64) + 0.5) * scale - 0.5
    lower_unclipped = torch.floor(source).long()
    upper_unclipped = lower_unclipped + 1
    upper_weight = source - lower_unclipped.to(source.dtype)
    lower_weight = 1.0 - upper_weight
    lower = lower_unclipped.clamp(0, input_size - 1)
    upper = upper_unclipped.clamp(0, input_size - 1)

    weights = torch.zeros(output_size, input_size, dtype=torch.float64)
    rows = torch.arange(output_size)
    weights[rows, lower] += lower_weight
    weights[rows, upper] += upper_weight
    weights = weights.to(device=device, dtype=dtype)
    _RESIZE_WEIGHT_CACHE[key] = weights
    return weights


def resize_bilinear(x: torch.Tensor, size) -> torch.Tensor:
    """Bilinear resize with a deterministic CUDA backward when requested."""
    output_h, output_w = int(size[0]), int(size[1])
    if not torch.are_deterministic_algorithms_enabled():
        return F.interpolate(x, size=(output_h, output_w), mode="bilinear", align_corners=False)

    input_h, input_w = x.shape[-2:]
    weight_h = _linear_resize_weights(input_h, output_h, x.device, x.dtype)
    weight_w = _linear_resize_weights(input_w, output_w, x.device, x.dtype)
    flat = x.reshape(-1, input_h, input_w)
    resized_h = torch.matmul(weight_h.unsqueeze(0), flat)
    resized = torch.matmul(resized_h, weight_w.t().unsqueeze(0))
    return resized.reshape(*x.shape[:-2], output_h, output_w)


class MultiStageUNetDecoder(nn.Module):
    """
    U-Net/FPN-style decoder for a list of features ordered shallow -> deep.

    For true hierarchical features, e.g. SpaVista, spatial sizes are typically different.
    For ViT last-block features, spatial sizes are usually the same. The same fusion code still works.
    """

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int,
        decoder_dim: int = 128,
        image_size: int = 224,
        num_post_blocks: int = 2,
        norm: str = "gn",
        use_deep_supervision: bool = False,
    ):
        super().__init__()
        if len(in_channels_list) < 1:
            raise ValueError("in_channels_list must be non-empty")
        self.in_channels_list = [int(c) for c in in_channels_list]
        self.out_channels = int(out_channels)
        self.decoder_dim = int(decoder_dim)
        self.image_size = int(image_size)
        self.num_post_blocks = int(num_post_blocks)
        self.norm = str(norm)
        self.use_deep_supervision = bool(use_deep_supervision)

        self.adapters = nn.ModuleList([
            nn.Conv2d(c, self.decoder_dim, kernel_size=1)
            for c in self.in_channels_list
        ])

        # One fusion block for each skip connection, deep -> shallow.
        self.fuse_blocks = nn.ModuleList([
            NormActConv(self.decoder_dim * 2, self.decoder_dim, norm=self.norm)
            for _ in range(max(0, len(self.in_channels_list) - 1))
        ])

        self.post_blocks = nn.Sequential(*[
            NormActConv(self.decoder_dim, self.decoder_dim, norm=self.norm)
            for _ in range(self.num_post_blocks)
        ])
        self.out_conv = nn.Conv2d(self.decoder_dim, self.out_channels, kernel_size=1)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        if len(feats) != len(self.adapters):
            raise RuntimeError(f"Expected {len(self.adapters)} feature maps, got {len(feats)}")

        # Convert all levels to a common decoder channel dim.
        adapted = [adapter(f.float()) for adapter, f in zip(self.adapters, feats)]

        # Start from deepest feature.
        x = adapted[-1]
        fuse_idx = 0
        for level_idx in range(len(adapted) - 2, -1, -1):
            skip = adapted[level_idx]
            if x.shape[-2:] != skip.shape[-2:]:
                x = resize_bilinear(x, skip.shape[-2:])
            x = torch.cat([x, skip], dim=1)
            x = self.fuse_blocks[fuse_idx](x)
            fuse_idx += 1

        if x.shape[-2:] != (self.image_size, self.image_size):
            x = resize_bilinear(x, (self.image_size, self.image_size))
        x = self.post_blocks(x)
        x = self.out_conv(x)
        return torch.sigmoid(x)

    def structure_string(self):
        return (
            f"MultiStageUNetDecoder(levels={len(self.in_channels_list)}, "
            f"in_channels={self.in_channels_list}, decoder_dim={self.decoder_dim}, "
            f"image_size={self.image_size}, post_blocks={self.num_post_blocks}, norm={self.norm})"
        )


class FrozenMultiStageEncoderUNet(nn.Module):
    def __init__(self, spatial_encoder: MultiStageFeatureExtractor, decoder: MultiStageUNetDecoder):
        super().__init__()
        self.spatial_encoder = spatial_encoder
        self.decoder = decoder
        for p in self.spatial_encoder.parameters():
            p.requires_grad = False
        self.spatial_encoder.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.spatial_encoder.eval()
        feats = self.spatial_encoder(x)
        return self.decoder(feats)


# =============================================================================
# Metrics
# =============================================================================
def _init_accumulators(n_channels: int) -> Dict[str, np.ndarray]:
    return {
        "sum_true": np.zeros(n_channels, dtype=np.float64),
        "sum_pred": np.zeros(n_channels, dtype=np.float64),
        "sum_true2": np.zeros(n_channels, dtype=np.float64),
        "sum_pred2": np.zeros(n_channels, dtype=np.float64),
        "sum_true_pred": np.zeros(n_channels, dtype=np.float64),
        "sum_abs_err": np.zeros(n_channels, dtype=np.float64),
        "sum_sq_err": np.zeros(n_channels, dtype=np.float64),
        "n_pix": np.zeros(n_channels, dtype=np.float64),
        "ssim_sum": np.zeros(n_channels, dtype=np.float64),
        "ssim_count": np.zeros(n_channels, dtype=np.float64),
    }


def _global_ssim_per_image_channel(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Simple global SSIM per image/channel.
    Inputs: [B,C,H,W] in [0,1]. Returns [B,C].
    This is not windowed SSIM, but is stable and dependency-free.
    """
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    dims = (2, 3)
    mux = y_true.mean(dim=dims)
    muy = y_pred.mean(dim=dims)
    vx = ((y_true - mux[:, :, None, None]) ** 2).mean(dim=dims)
    vy = ((y_pred - muy[:, :, None, None]) ** 2).mean(dim=dims)
    cov = ((y_true - mux[:, :, None, None]) * (y_pred - muy[:, :, None, None])).mean(dim=dims)
    ssim = ((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux ** 2 + muy ** 2 + c1) * (vx + vy + c2) + 1e-12)
    return ssim


def update_metrics_acc(acc: Dict[str, np.ndarray], y_true: torch.Tensor, y_pred: torch.Tensor):
    y_true = y_true.detach().float().cpu().clamp(0, 1)
    y_pred = y_pred.detach().float().cpu().clamp(0, 1)
    _, c, _, _ = y_true.shape
    for j in range(c):
        t = y_true[:, j].reshape(-1).numpy().astype(np.float64)
        p = y_pred[:, j].reshape(-1).numpy().astype(np.float64)
        acc["sum_true"][j] += t.sum()
        acc["sum_pred"][j] += p.sum()
        acc["sum_true2"][j] += (t * t).sum()
        acc["sum_pred2"][j] += (p * p).sum()
        acc["sum_true_pred"][j] += (t * p).sum()
        err = p - t
        acc["sum_abs_err"][j] += np.abs(err).sum()
        acc["sum_sq_err"][j] += (err * err).sum()
        acc["n_pix"][j] += t.size

    ssim = _global_ssim_per_image_channel(y_true, y_pred).numpy().astype(np.float64)
    acc["ssim_sum"] += np.nan_to_num(ssim, nan=0.0).sum(axis=0)
    acc["ssim_count"] += np.isfinite(ssim).sum(axis=0)


def finalize_metrics(acc: Dict[str, np.ndarray], channel_names: List[str], model_name: str, split_name: str):
    rows = []
    for j, ch in enumerate(channel_names):
        n = acc["n_pix"][j]
        mse = acc["sum_sq_err"][j] / max(n, 1.0)
        rmse = float(np.sqrt(mse))
        mae = acc["sum_abs_err"][j] / max(n, 1.0)
        true_mean = acc["sum_true"][j] / max(n, 1.0)
        pred_mean = acc["sum_pred"][j] / max(n, 1.0)
        true_var = acc["sum_true2"][j] / max(n, 1.0) - true_mean ** 2
        pred_var = acc["sum_pred2"][j] / max(n, 1.0) - pred_mean ** 2
        cov = acc["sum_true_pred"][j] / max(n, 1.0) - true_mean * pred_mean
        pcc = cov / (np.sqrt(max(true_var, 1e-12)) * np.sqrt(max(pred_var, 1e-12)))
        psnr = 10.0 * np.log10(1.0 / max(mse, 1e-12))
        ssim = acc["ssim_sum"][j] / max(acc["ssim_count"][j], 1.0)
        rows.append({
            "model": model_name,
            "split": split_name,
            "channel": ch,
            "pixel_mse": float(mse),
            "pixel_rmse": float(rmse),
            "pixel_mae": float(mae),
            "pixel_pcc": float(pcc),
            "psnr": float(psnr),
            "global_ssim": float(ssim),
            "true_mean": float(true_mean),
            "pred_mean": float(pred_mean),
            "true_std": float(np.sqrt(max(true_var, 0.0))),
            "pred_std": float(np.sqrt(max(pred_var, 0.0))),
        })
    per_df = pd.DataFrame(rows)
    overall = {
        "model": model_name,
        "split": split_name,
        "n_channels": int(len(channel_names)),
        "mean_pixel_mse": float(per_df["pixel_mse"].mean()),
        "mean_pixel_rmse": float(per_df["pixel_rmse"].mean()),
        "mean_pixel_mae": float(per_df["pixel_mae"].mean()),
        "mean_pixel_pcc": float(per_df["pixel_pcc"].mean()),
        "mean_psnr": float(per_df["psnr"].mean()),
        "mean_global_ssim": float(per_df["global_ssim"].mean()),
        "median_pixel_mse": float(per_df["pixel_mse"].median()),
        "median_pixel_rmse": float(per_df["pixel_rmse"].median()),
        "median_pixel_mae": float(per_df["pixel_mae"].median()),
        "median_pixel_pcc": float(per_df["pixel_pcc"].median()),
        "median_psnr": float(per_df["psnr"].median()),
        "median_global_ssim": float(per_df["global_ssim"].median()),
    }
    return overall, per_df


# =============================================================================
# Training / evaluation
# =============================================================================
def make_loader(dataset: Dataset, args, train: bool):
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    return DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=bool(train),
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=int(args.num_workers) > 0,
    )


def make_loss(pred: torch.Tensor, target: torch.Tensor, args):
    if args.loss == "l1":
        return F.l1_loss(pred, target)
    if args.loss == "mse":
        return F.mse_loss(pred, target)
    if args.loss == "l1_mse":
        return args.l1_weight * F.l1_loss(pred, target) + args.mse_weight * F.mse_loss(pred, target)
    raise ValueError("--loss must be l1, mse, or l1_mse")


@torch.no_grad()
def evaluate_model(
    model: FrozenMultiStageEncoderUNet,
    loader: DataLoader,
    args,
    device,
    channel_names: List[str],
    model_name: str,
    split_name: str,
    save_dir: Optional[str] = None,
    save_examples: bool = False,
):
    model.to(device)
    model.eval()
    acc = _init_accumulators(len(channel_names))
    total_loss = 0.0
    seen = 0

    example_rows = []
    example_saved = 0
    example_dir = None
    if save_dir is not None and save_examples:
        example_dir = os.path.join(save_dir, f"{split_name}_examples")
        ensure_dir(example_dir)

    for he, target, indices in tqdm(loader, desc=f"[{model_name}] Eval {split_name}"):
        he = he.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with autocast_context(args, device):
            pred = model(he)
            loss = make_loss(pred.float(), target.float(), args)
        update_metrics_acc(acc, target, pred)
        bsz = int(he.shape[0])
        total_loss += float(loss.detach().cpu().item()) * bsz
        seen += bsz

        if example_dir is not None and example_saved < int(args.save_examples):
            n_take = min(int(args.save_examples) - example_saved, bsz)
            for k in range(n_take):
                he_img = tensor_to_numpy_img(denormalize_he(he[k].detach().cpu()))
                true_img = tensor_to_numpy_img(target[k].detach().cpu())
                pred_img = tensor_to_numpy_img(pred[k].detach().cpu())
                err = torch.abs(pred[k].detach().cpu().float() - target[k].detach().cpu().float()).mean(dim=0, keepdim=True)
                err_img = tensor_to_numpy_img(err / (err.max() + 1e-8))
                panel_path = os.path.join(example_dir, f"example_{example_saved:03d}.png")
                save_example_panel(he_img, true_img, pred_img, err_img, panel_path, channel_names)
                example_rows.append({"example_id": example_saved, "panel_path": panel_path})
                example_saved += 1

    overall, per_df = finalize_metrics(acc, channel_names, model_name, split_name)
    overall["loss"] = total_loss / max(seen, 1)
    overall["n_images"] = int(seen)

    if save_dir is not None:
        pd.DataFrame([overall]).to_csv(os.path.join(save_dir, f"{split_name}_metrics_overall.csv"), index=False)
        per_df.to_csv(os.path.join(save_dir, f"{split_name}_metrics_per_channel.csv"), index=False)
        if example_rows:
            pd.DataFrame(example_rows).to_csv(os.path.join(example_dir, "examples.csv"), index=False)

    return overall, per_df


def save_example_panel(he_img: np.ndarray, true_img: np.ndarray, pred_img: np.ndarray, err_img: np.ndarray, out_path: str, channel_names: List[str]):
    titles = ["H&E", "True mIF", "Predicted mIF", "Mean abs. error"]
    imgs = [he_img, true_img, pred_img, err_img]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle("Channels: " + ", ".join(channel_names), fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def train_one_model(
    model: FrozenMultiStageEncoderUNet,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args,
    device,
    channel_names: List[str],
    save_dir: str,
):
    ensure_dir(save_dir)
    model.to(device)
    model.spatial_encoder.eval()
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma) if args.lr_step_size > 0 else None
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.amp == "fp16"))

    best_score = -1e18
    best_epoch = -1
    history = []

    for epoch in range(1, args.max_epoch + 1):
        model.train()
        model.spatial_encoder.eval()
        model.decoder.train()
        running_loss = 0.0
        seen = 0
        pbar = tqdm(train_loader, desc=f"[{model_name}] Train U-Net decoder {epoch}/{args.max_epoch}")
        for he, target, _ in pbar:
            he = he.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(args, device):
                pred = model(he)
                loss = make_loss(pred.float(), target.float(), args)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if args.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), args.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), args.grad_clip_norm)
                optimizer.step()

            bsz = int(he.shape[0])
            running_loss += float(loss.detach().cpu().item()) * bsz
            seen += bsz
            pbar.set_postfix(loss=running_loss / max(seen, 1))

        if scheduler is not None:
            scheduler.step()

        row = {"epoch": epoch, "train_loss": running_loss / max(seen, 1), "lr": float(optimizer.param_groups[0]["lr"])}

        if args.eval_every_epoch > 0 and (epoch % int(args.eval_every_epoch) == 0 or epoch == args.max_epoch):
            val_overall, val_per_channel = evaluate_model(model, val_loader, args, device, channel_names, model_name, "val", save_dir=None)
            for k, v in val_overall.items():
                if isinstance(v, (int, float, np.integer, np.floating)):
                    row[f"val_{k}"] = float(v)

            min_channel_pcc = float(np.nanmin(val_per_channel["pixel_pcc"].to_numpy(dtype=float)))
            row["val_min_channel_pcc"] = min_channel_pcc

            if args.best_metric == "val_min_channel_pcc":
                score = min_channel_pcc
            elif args.best_metric == "val_mean_pcc":
                score = float(val_overall["mean_pixel_pcc"])
            elif args.best_metric == "val_median_pcc":
                score = float(val_overall["median_pixel_pcc"])
            elif args.best_metric == "val_mean_ssim":
                score = float(val_overall["mean_global_ssim"])
            elif args.best_metric == "val_neg_loss":
                score = -float(val_overall["loss"])
            elif args.best_metric == "val_neg_mae":
                score = -float(val_overall["mean_pixel_mae"])
            else:
                raise ValueError(f"Unknown best_metric={args.best_metric}")

            print(
                f"[VAL] {model_name} epoch={epoch:03d}, "
                f"median_pcc={val_overall['median_pixel_pcc']:.6f}, "
                f"mean_pcc={val_overall['mean_pixel_pcc']:.6f}, "
                f"mean_ssim={val_overall['mean_global_ssim']:.6f}, "
                f"mae={val_overall['mean_pixel_mae']:.6f}, "
                f"loss={val_overall['loss']:.6f}, score={score:.6f}"
            )

            if score > best_score:
                best_score = score
                best_epoch = epoch
                torch.save({
                    "model_name": model_name,
                    "channel_names": channel_names,
                    "decoder_structure": model.decoder.structure_string(),
                    "best_epoch": int(best_epoch),
                    "best_score": float(best_score),
                    "best_metric": args.best_metric,
                    "decoder_state_dict": {k: v.detach().cpu() for k, v in model.decoder.state_dict().items()},
                    "args": vars(args),
                }, os.path.join(save_dir, "best_decoder.pt"))
                print(f"[SAVE] New best U-Net decoder saved at epoch={epoch}, score={score:.6f}")

        history.append(row)
        print(f"[TRAIN] {model_name} epoch={epoch:03d}, loss={row['train_loss']:.6f}")

    torch.save({
        "model_name": model_name,
        "channel_names": channel_names,
        "decoder_structure": model.decoder.structure_string(),
        "last_epoch": int(args.max_epoch),
        "decoder_state_dict": {k: v.detach().cpu() for k, v in model.decoder.state_dict().items()},
        "args": vars(args),
    }, os.path.join(save_dir, "last_decoder.pt"))

    pd.DataFrame(history).to_csv(os.path.join(save_dir, "train_log.csv"), index=False)
    return best_epoch, best_score


def make_dummy_input(args, device) -> torch.Tensor:
    dummy = torch.zeros(1, 3, args.image_size, args.image_size, device=device)
    if args.normal:
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        dummy = (dummy - mean) / std
    return dummy


def infer_feature_shapes(spatial_encoder: MultiStageFeatureExtractor, args, device) -> List[Tuple[int, int, int]]:
    spatial_encoder.to(device)
    spatial_encoder.eval()
    dummy = make_dummy_input(args, device)
    with torch.no_grad():
        feats = spatial_encoder(dummy)
    if not isinstance(feats, list) or len(feats) == 0:
        raise RuntimeError("Expected non-empty list of feature maps.")
    shapes = []
    for f in feats:
        if f.dim() != 4:
            raise RuntimeError(f"Expected feature map [B,C,H,W], got {tuple(f.shape)}")
        shapes.append((int(f.shape[1]), int(f.shape[2]), int(f.shape[3])))
    return shapes


def load_best_decoder(model: FrozenMultiStageEncoderUNet, save_dir: str, device):
    ckpt_path = os.path.join(save_dir, "best_decoder.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(save_dir, "last_decoder.pt")
    check_file(ckpt_path, "decoder checkpoint")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.decoder.load_state_dict(ckpt["decoder_state_dict"], strict=True)
    model.to(device)
    model.eval()
    print(f"[LOAD] Loaded decoder checkpoint: {ckpt_path}")
    return ckpt_path


def run_one_model(model_key: str, datasets: Dict[str, Dataset], args, device, output_dir: str, channel_names: List[str]):
    encoder, _, canonical_name = build_encoder(args, model_key)
    model_dir = os.path.join(output_dir, sanitize_name(canonical_name))
    ensure_dir(model_dir)

    spatial_encoder = MultiStageFeatureExtractor(
        encoder=encoder,
        model_name=canonical_name,
        num_feature_levels=args.num_feature_levels,
    )
    feat_shapes = infer_feature_shapes(spatial_encoder, args, device)
    in_channels_list = [s[0] for s in feat_shapes]

    print("=" * 80)
    print(f"[MODEL] {canonical_name}")
    print(f"[FEATURE LEVELS] shallow -> deep, count={len(feat_shapes)}")
    for i, (c, h, w) in enumerate(feat_shapes):
        print(f"  level_{i}: C={c}, HxW={h}x{w}")
    print(f"[OUTPUT CHANNELS] {channel_names}")
    print("=" * 80)

    decoder = MultiStageUNetDecoder(
        in_channels_list=in_channels_list,
        out_channels=len(channel_names),
        decoder_dim=args.decoder_dim,
        image_size=args.image_size,
        num_post_blocks=args.decoder_post_blocks,
        norm=args.decoder_norm,
    )
    model = FrozenMultiStageEncoderUNet(spatial_encoder, decoder)
    print(f"[DECODER] {decoder.structure_string()}")

    train_loader = make_loader(datasets["train"], args, train=True)
    val_loader = make_loader(datasets["val"], args, train=False)
    test_loader = make_loader(datasets["test"], args, train=False)

    if args.eval_only:
        checkpoint_path = os.path.join(model_dir, "best_decoder.pt")
        check_file(checkpoint_path, "best decoder checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        previous_args = checkpoint.get("args", {})
        for key in ("out_channels", "image_size", "num_feature_levels", "decoder_dim", "decoder_post_blocks", "decoder_norm", "normal", "best_metric"):
            if key in previous_args and previous_args[key] != getattr(args, key):
                raise ValueError(f"Checkpoint argument mismatch for {key}: {previous_args[key]} != {getattr(args, key)}")
        best_epoch = int(checkpoint["best_epoch"])
        best_score = float(checkpoint["best_score"])
        print(f"[EVAL ONLY] Using best decoder from epoch={best_epoch}, score={best_score:.6f}")
    else:
        best_epoch, best_score = train_one_model(model, canonical_name, train_loader, val_loader, args, device, channel_names, model_dir)
    load_best_decoder(model, model_dir, device)

    val_overall, _ = evaluate_model(model, val_loader, args, device, channel_names, canonical_name, "val", save_dir=model_dir, save_examples=False)
    test_overall, test_per_df = evaluate_model(model, test_loader, args, device, channel_names, canonical_name, "test", save_dir=model_dir, save_examples=True)

    test_overall["best_epoch"] = int(best_epoch)
    test_overall["best_score"] = float(best_score)
    test_overall["best_metric"] = args.best_metric
    test_overall["decoder_dim"] = int(args.decoder_dim)
    test_overall["decoder_post_blocks"] = int(args.decoder_post_blocks)
    test_overall["decoder_norm"] = args.decoder_norm
    test_overall["num_feature_levels"] = int(args.num_feature_levels)
    test_overall["feature_shapes"] = ";".join([f"{c}x{h}x{w}" for c, h, w in feat_shapes])

    pd.DataFrame([test_overall]).to_csv(os.path.join(model_dir, "test_metrics_overall.csv"), index=False)
    test_per_df.to_csv(os.path.join(model_dir, "test_metrics_per_channel.csv"), index=False)

    print("\n" + "=" * 80)
    print(f"[RESULT] {canonical_name}")
    for k in ["median_pixel_pcc", "mean_pixel_pcc", "mean_global_ssim", "mean_psnr", "mean_pixel_mae", "best_epoch"]:
        print(f"{k}: {test_overall[k]}")
    print("=" * 80)

    del model, encoder, spatial_encoder, decoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return test_overall, test_per_df


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="HEMIT H&E -> mIF reconstruction using multi-stage SpaVista features + U-Net decoder."
    )

    # Dataset
    parser.add_argument("--config", type=str, default="configs/paths.yaml", help="YAML file containing checkpoint and dataset paths.")
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--data_root", type=str, default="", help="Root containing HEMIT train/val/test directories.")
    parser.add_argument("--out_channels", type=str, default="panCK,CD3", help="Use panCK,CD3 or panCK,CD3,DAPI")
    parser.add_argument("--image_size", type=int, default=224, help="Resize H&E and mIF to this size.")
    parser.add_argument("--normal", type=str2bool, default=True)
    parser.add_argument("--strict_path_check", type=str2bool, default=True)

    # Models
    parser.add_argument("--project_root", type=str, default="")
    parser.add_argument("--spavista_ckpt", type=str, default=None)
    parser.add_argument("--spavista_arch", type=str, default="mamba_vision_base")

    parser.add_argument("--strict_load", type=str2bool, default=False)

    # Multi-stage feature extraction
    parser.add_argument("--num_feature_levels", type=int, default=4, help="SpaVista: use N unique pyramid levels; ViT FMs: use last N block outputs.")

    # Decoder and training
    parser.add_argument("--decoder_dim", type=int, default=128)
    parser.add_argument("--decoder_post_blocks", type=int, default=2)
    parser.add_argument("--decoder_norm", type=str, default="gn", choices=["gn", "bn", "none"])
    parser.add_argument("--loss", type=str, default="l1_mse", choices=["l1", "mse", "l1_mse"])
    parser.add_argument("--l1_weight", type=float, default=1.0)
    parser.add_argument("--mse_weight", type=float, default=0.2)
    parser.add_argument("--max_epoch", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr_step_size", type=int, default=20)
    parser.add_argument("--lr_gamma", type=float, default=0.9)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--eval_every_epoch", type=int, default=1)
    parser.add_argument("--best_metric", type=str, default="val_mean_ssim", choices=["val_min_channel_pcc", "val_median_pcc", "val_mean_pcc", "val_mean_ssim", "val_neg_loss", "val_neg_mae"])

    # Runtime
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--tag", type=str, default="hemit_mif_spavista_bs6_10epoch_mean_ssim")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size used for the reported SpaVista HEMIT result.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_examples", type=int, default=12)
    parser.add_argument("--eval_only", action="store_true", help="Evaluate an existing best_decoder.pt without training.")
    parser.add_argument("--device_num", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--amp", type=str, default="fp16", choices=["none", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument(
        "--deterministic",
        type=str2bool,
        default=True,
        help="Use deterministic PyTorch/CUDA algorithms for reproducible training.",
    )
    parser.add_argument(
        "--allow_tf32",
        type=str2bool,
        default=False,
        help="Allow TF32 on supported GPUs. Disabled by default for numerical reproducibility.",
    )

    args = parser.parse_args()
    config = load_path_config(args.config)
    task_config = config.get("hemit", {}) or {}
    args.csv_path = args.csv_path or task_config.get("labels_csv")
    args.data_root = args.data_root or task_config.get("root") or ""
    args.spavista_ckpt = args.spavista_ckpt or config.get("checkpoint")
    if not args.csv_path:
        parser.error("Set hemit.labels_csv in --config or pass --csv_path.")
    if not args.data_root:
        parser.error("Set hemit.root in --config or pass --data_root.")
    if not args.spavista_ckpt:
        parser.error("Set checkpoint in --config or pass --spavista_ckpt.")

    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
    seed_torch(args.seed)
    add_project_root(args.project_root)
    device = get_device(args)

    if args.output_dir is None or str(args.output_dir).strip() == "":
        args.output_dir = os.path.join("results", args.tag)
    ensure_dir(args.output_dir)
    if not args.eval_only:
        save_run_config(args, args.output_dir)

    out_channel_list = parse_comma_list(args.out_channels)
    _, channel_names = channel_indices_from_names(out_channel_list)
    model_keys = ["spavista"]

    print("=" * 80)
    print("HEMIT H&E to mIF reconstruction")
    print("Multi-stage / multi-layer frozen FM features -> U-Net/FPN decoder -> mIF image")
    print(f"csv_path:              {args.csv_path}")
    print(f"model:                 {model_keys[0]}")
    print(f"out_channels:          {channel_names}")
    print(f"image_size:            {args.image_size}")
    print(f"num_feature_levels:    {args.num_feature_levels}")
    print(f"decoder_dim:           {args.decoder_dim}")
    print(f"decoder_post_blocks:   {args.decoder_post_blocks}")
    print(f"decoder_norm:          {args.decoder_norm}")
    print(f"loss:                  {args.loss}")
    print(f"best_metric:           {args.best_metric}")
    print(f"batch_size:            {args.batch_size}")
    print(f"device:                {device}")
    print("=" * 80)

    datasets = {
        "train": HEMITImageReconstructionDataset(args.csv_path, "train", out_channels=channel_names, data_root=args.data_root, image_size=args.image_size, normal=args.normal, strict_path_check=args.strict_path_check, verbose=True),
        "val": HEMITImageReconstructionDataset(args.csv_path, "val", out_channels=channel_names, data_root=args.data_root, image_size=args.image_size, normal=args.normal, strict_path_check=args.strict_path_check, verbose=True),
        "test": HEMITImageReconstructionDataset(args.csv_path, "test", out_channels=channel_names, data_root=args.data_root, image_size=args.image_size, normal=args.normal, strict_path_check=args.strict_path_check, verbose=True),
    }

    all_overall = []
    all_per_channel = []

    for model_key in model_keys:
        overall, per_df = run_one_model(model_key, datasets, args, device, args.output_dir, channel_names)
        all_overall.append(overall)
        all_per_channel.append(per_df)

    overall_df = pd.DataFrame(all_overall)
    per_channel_df = pd.concat(all_per_channel, axis=0, ignore_index=True)

    overall_path = os.path.join(args.output_dir, "all_model_test_metrics.csv")
    per_channel_path = os.path.join(args.output_dir, "all_model_channel_metrics.csv")
    overall_df.to_csv(overall_path, index=False)
    per_channel_df.to_csv(per_channel_path, index=False)

    print("\n" + "=" * 80)
    print("[DONE] Saved summary:")
    print(overall_path)
    print(per_channel_path)
    show_cols = [
        "model", "median_pixel_pcc", "mean_pixel_pcc", "mean_global_ssim", "mean_psnr",
        "mean_pixel_mae", "median_pixel_mae", "best_epoch", "feature_shapes",
    ]
    show_cols = [c for c in show_cols if c in overall_df.columns]
    print(overall_df[show_cols].to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
