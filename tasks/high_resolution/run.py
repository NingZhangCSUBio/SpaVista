#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train and evaluate a frozen SpaVista encoder for Visium HD
cross-resolution gene expression prediction.

Default task:
    Train: 16 um VisiumHD bins, image patch -> expression
    Test:   8 um VisiumHD bins, image patch -> expression

Design:
    1) No PCA and no Ridge.
    2) The FM encoder is frozen.
    3) Only a two-layer Linear gene head is trained:
           encoder_feature_dim -> 1024 -> n_genes
    4) Only the Linear head is saved.
    5) Dataset supports:
           --normal     : ImageNet normalization on/off
    6) Cached frozen-encoder features are saved as float16 by default
       to reduce disk usage, and converted back to float32 during head training.
    7) The test set is evaluated once, after the final training epoch.

Expected VisiumHD folder structure:
    <base_dir>/<train_slide>/exp_HVG200_MARKER10.csv
    <base_dir>/<train_slide>/subposition.parquet
    <base_dir>/<test_slide>/exp_HVG200_MARKER10.csv
    <base_dir>/<test_slide>/subposition.parquet

The H&E image is provided separately by --train_he_tif and optionally --test_he_tif.
If --test_he_tif is empty, it reuses --train_he_tif.

Run this module from the SpaVista repository root.
"""

import argparse
import json
import os
import random
import sys
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm import tqdm

from spavista.config import load_path_config

try:
    import scprep as scp
except Exception as e:
    raise ImportError(
        "This script needs scprep because the original VisiumHD dataset uses "
        "library_size_normalize + log transform. Install it with: pip install scprep"
    ) from e

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None


ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
# =============================================================================
# Basic utils
# =============================================================================
def str2bool(v):
    """argparse bool parser. Supports True/False, yes/no, 1/0."""
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in ("yes", "true", "t", "y", "1"):
        return True
    if v in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def seed_torch(seed: int = 2024):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def check_file(path: str, name: str):
    if path is None or str(path).strip() == "":
        raise ValueError(f"{name} is empty.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find {name}: {path}")


def save_run_config(args, save_path: str):
    ensure_dir(save_path)
    args_dict = vars(args).copy()
    run_info = {
        "saved_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version,
        "command": " ".join(sys.argv),
        "args": args_dict,
    }
    json_path = os.path.join(save_path, "run_config.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=4, ensure_ascii=False)

    command_path = os.path.join(save_path, "run_command.txt")
    with open(command_path, "w", encoding="utf-8") as f:
        f.write(" ".join(sys.argv) + "\n")

    print(f"[INFO] Run config saved to: {json_path}")
    print(f"[INFO] Run command saved to: {command_path}")


def add_project_root(project_root: str):
    if project_root and str(project_root).strip() != "":
        project_root = os.path.abspath(project_root)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        print(f"[INFO] Added project_root to sys.path: {project_root}")


def make_auto_tag(args) -> str:
    task_name = "HVG200" if args.target_task == "hvg200" else "10markers"
    return f"visiumhd/SpaVista_{task_name}"


def apply_target_task_preset(args):
    """Apply the exact downstream settings used by the two reference experiments."""
    marker_genes = "ERBB2,CD24,SCGB2A2,KRT5,KRT14,ACTA2,MMP11,VWF,MS4A1,ADIPOQ"
    shared = {
        "expression_norm_scope": "target",
        "crop_radius": 112,
        "normal": True,
        "head_hidden_dim": 1024,
        "feature_cache_dtype": "float16",
        "feature_batch_size": 2048,
        "head_batch_size": 64,
        "feature_num_workers": 8,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "lr_step_size": 50,
        "lr_gamma": 0.9,
        "grad_clip_norm": 0.0,
        "amp": "fp16",
        "seed": 2024,
    }
    for name, value in shared.items():
        setattr(args, name, value)

    args.marker_gene_file = ""
    args.marker_from_hvg_tail = 0
    if args.target_task == "hvg200":
        args.gene_mode = "first_k"
        args.gene_num = 200
        args.marker_genes = ""
        args.max_epoch = 50
    elif args.target_task == "marker10":
        args.gene_mode = "markers_only"
        args.gene_num = 0
        args.marker_genes = marker_genes
        args.max_epoch = 25
    else:
        raise ValueError(f"Unsupported target_task: {args.target_task}")


def get_device(args) -> torch.device:
    if args.device and str(args.device).strip() != "":
        return torch.device(args.device)
    if torch.cuda.is_available():
        return torch.device(f"cuda:{args.device_num}")
    return torch.device("cpu")


def get_autocast_dtype(args, device: torch.device):
    if device.type != "cuda":
        return None
    if args.amp == "fp16":
        return torch.float16
    if args.amp == "bf16":
        return torch.bfloat16
    return None


def autocast_context(args, device: torch.device):
    dtype = get_autocast_dtype(args, device)
    if device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=dtype is not None, dtype=dtype)
    import contextlib
    return contextlib.nullcontext()


def resolve_feature_cache_dtype(args):
    """Return numpy dtype used for cached frozen-encoder features."""
    dtype = str(getattr(args, "feature_cache_dtype", "float16")).lower()
    if dtype in ("float16", "fp16", "half"):
        return np.float16
    if dtype in ("float32", "fp32", "single"):
        return np.float32
    raise ValueError(f"Unsupported feature_cache_dtype={dtype}. Choose float16 or float32.")


def _unique_preserve_order(items: List[str]) -> List[str]:
    """Deduplicate a list while preserving the original order."""
    out = []
    seen = set()
    for x in items:
        x = str(x)
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _read_gene_names_from_file(path: str) -> List[str]:
    """
    Read gene names from .npy / .txt / .csv / .tsv.

    Rules:
        .npy: 1D array of gene names.
        .txt: one gene per line.
        .csv/.tsv: if there is one column, use it; otherwise use the first column.
    """
    if path is None or str(path).strip() == "":
        return []
    check_file(path, "marker_gene_file")
    path = str(path)

    if path.lower().endswith(".npy"):
        arr = np.load(path, allow_pickle=True)
        if arr.ndim != 1:
            raise ValueError(f"Gene .npy file should be 1D, got shape={arr.shape}: {path}")
        genes = [str(g) for g in list(arr)]
    elif path.lower().endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            genes = [line.strip() for line in f if line.strip()]
    elif path.lower().endswith(".csv"):
        df = pd.read_csv(path)
        genes = df.iloc[:, 0].astype(str).tolist()
    elif path.lower().endswith(".tsv"):
        df = pd.read_csv(path, sep="\t")
        genes = df.iloc[:, 0].astype(str).tolist()
    else:
        raise ValueError(f"Unsupported gene file format: {path}")

    return _unique_preserve_order([g for g in genes if str(g).strip() != ""])


def _parse_marker_genes(
    base_gene_set: List[str],
    marker_genes: str = "",
    marker_gene_file: str = "",
    marker_from_hvg_tail: int = 0,
) -> List[str]:
    """
    Resolve marker genes from:
        1) --marker_genes comma-separated genes
        2) --marker_gene_file
        3) --marker_from_hvg_tail N, using the last N genes in hvg_file
    """
    markers = []

    if marker_genes is not None and str(marker_genes).strip() != "":
        markers.extend([g.strip() for g in str(marker_genes).split(",") if g.strip()])

    if marker_gene_file is not None and str(marker_gene_file).strip() != "":
        markers.extend(_read_gene_names_from_file(marker_gene_file))

    marker_from_hvg_tail = int(marker_from_hvg_tail)
    if marker_from_hvg_tail > 0:
        if marker_from_hvg_tail > len(base_gene_set):
            raise ValueError(
                f"--marker_from_hvg_tail={marker_from_hvg_tail} is larger than hvg_file gene number={len(base_gene_set)}"
            )
        markers.extend(base_gene_set[-marker_from_hvg_tail:])

    return _unique_preserve_order(markers)


def select_target_genes_from_hvg_file(
    hvg_file: str,
    gene_mode: str = "all",
    gene_num: int = 0,
    marker_genes: str = "",
    marker_gene_file: str = "",
    marker_from_hvg_tail: int = 0,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Select target genes for VisiumHD expression prediction.

    gene_mode:
        all:
            Use all genes from hvg_file. Original behavior.

        first_k:
            Use the first --gene_num genes from hvg_file.

        topk_plus_markers:
            First remove markers from hvg_file, then take the first --gene_num
            remaining genes as HVGs, and append markers.
            Example: --gene_mode topk_plus_markers --gene_num 100 --marker_from_hvg_tail 2

        exclude_markers:
            Use all genes from hvg_file except marker genes.
            Example: --gene_mode exclude_markers --marker_from_hvg_tail 2

        markers_only:
            Use marker genes only.
            Example: --gene_mode markers_only --marker_from_hvg_tail 2
    """
    gene_array = np.load(hvg_file, allow_pickle=True)
    if gene_array.ndim != 1:
        raise ValueError(f"hvg_file should be a 1D gene-name array, got shape={gene_array.shape}")

    base_gene_set = [str(g) for g in list(gene_array)]
    base_gene_set = _unique_preserve_order(base_gene_set)

    if len(base_gene_set) == 0:
        raise ValueError("hvg_file contains zero genes.")

    marker_gene_set = _parse_marker_genes(
        base_gene_set=base_gene_set,
        marker_genes=marker_genes,
        marker_gene_file=marker_gene_file,
        marker_from_hvg_tail=marker_from_hvg_tail,
    )

    gene_mode = str(gene_mode).lower()
    gene_num = int(gene_num)

    if gene_mode == "all":
        target_gene_set = list(base_gene_set)

    elif gene_mode == "first_k":
        if gene_num <= 0:
            raise ValueError("--gene_num must be > 0 when --gene_mode first_k.")
        target_gene_set = base_gene_set[:gene_num]

    elif gene_mode == "topk_plus_markers":
        if gene_num <= 0:
            raise ValueError("--gene_num must be > 0 when --gene_mode topk_plus_markers.")
        if len(marker_gene_set) == 0:
            warnings.warn(
                "--gene_mode topk_plus_markers was used, but no markers were provided. "
                "It will behave like first_k."
            )
        marker_set = set(marker_gene_set)
        hvg_candidates = [g for g in base_gene_set if g not in marker_set]
        target_gene_set = hvg_candidates[:gene_num] + marker_gene_set

    elif gene_mode == "exclude_markers":
        if len(marker_gene_set) == 0:
            warnings.warn(
                "--gene_mode exclude_markers was used, but no markers were provided. "
                "It will behave like all."
            )
        marker_set = set(marker_gene_set)
        target_gene_set = [g for g in base_gene_set if g not in marker_set]

    elif gene_mode == "markers_only":
        if len(marker_gene_set) == 0:
            raise ValueError("--gene_mode markers_only requires --marker_genes, --marker_gene_file, or --marker_from_hvg_tail.")
        target_gene_set = list(marker_gene_set)

    else:
        raise ValueError(
            f"Unsupported --gene_mode={gene_mode}. Choose from: "
            "all, first_k, topk_plus_markers, exclude_markers, markers_only."
        )

    target_gene_set = _unique_preserve_order(target_gene_set)

    missing_markers = [g for g in marker_gene_set if g not in base_gene_set]
    if len(missing_markers) > 0:
        warnings.warn(
            f"{len(missing_markers)} marker genes are not present in hvg_file. "
            f"They will still be checked against the configured expression CSV. Examples: {missing_markers[:10]}"
        )

    if len(target_gene_set) == 0:
        raise ValueError(
            f"Selected zero target genes. gene_mode={gene_mode}, gene_num={gene_num}, markers={marker_gene_set}"
        )

    return target_gene_set, base_gene_set, marker_gene_set


# =============================================================================
# VisiumHD Dataset
# =============================================================================
class ViT_VisiumHD(Dataset):
    """
    One slide/bin-level VisiumHD dataset.

    Returns:
        patch_tensor: (3, 224, 224), float32
        pixel_xy:     (2,), int32, [x, y] in full-resolution pixel space
        exp:          (G,), float32, log-library-size-normalized expression
    """

    REQUIRED_POS_COLS = ["barcode", "pxl_col_in_fullres", "pxl_row_in_fullres"]

    def __init__(
        self,
        base_dir: str,
        he_tif_path: str,
        hvg_file: str,
        slide_name: str,
        expression_file: str = "exp_HVG200_MARKER10.csv",
        r: int = 112,
        train: bool = False,
        train_aug: bool = False,
        normal: bool = True,
        check_dataset: bool = True,
        gene_mode: str = "all",
        gene_num: int = 0,
        marker_genes: str = "",
        marker_gene_file: str = "",
        marker_from_hvg_tail: int = 0,
        expression_norm_scope: str = "base",
    ):
        super().__init__()
        self.base_dir = base_dir
        self.he_tif_path = he_tif_path
        self.hvg_file = hvg_file
        self.slide_name = slide_name
        self.expression_file = expression_file
        self.r = int(r)
        self.train = bool(train)
        self.train_aug = bool(train_aug)
        self.normal = bool(normal)
        self.check_dataset = bool(check_dataset)

        self._check_basic_files()

        self.gene_mode = str(gene_mode).lower()
        self.gene_num = int(gene_num)
        self.marker_genes_arg = marker_genes
        self.marker_gene_file = marker_gene_file
        self.marker_from_hvg_tail = int(marker_from_hvg_tail)
        self.expression_norm_scope = str(expression_norm_scope).lower()

        self.gene_set, self.base_gene_set, self.marker_gene_set = select_target_genes_from_hvg_file(
            hvg_file=hvg_file,
            gene_mode=self.gene_mode,
            gene_num=self.gene_num,
            marker_genes=marker_genes,
            marker_gene_file=marker_gene_file,
            marker_from_hvg_tail=self.marker_from_hvg_tail,
        )

        if len(set(self.gene_set)) != len(self.gene_set):
            dup = pd.Series(self.gene_set).value_counts()
            dup = dup[dup > 1].index.tolist()[:10]
            raise ValueError(f"Selected gene_set contains duplicated gene names. Examples: {dup}")

        if self.expression_norm_scope not in ("target", "base"):
            raise ValueError("--expression_norm_scope must be 'target' or 'base'.")

        # If expression_norm_scope='base', expression is normalized over the full hvg_file gene set,
        # then subset to target genes. This keeps marker-gene scale consistent with the original
        # complete gene-panel setting and avoids normalizing marker-only targets over very few genes.
        if self.expression_norm_scope == "base":
            self._required_gene_set = _unique_preserve_order(self.base_gene_set + self.gene_set)
        else:
            self._required_gene_set = list(self.gene_set)

        meta = self.get_meta(slide_name)
        self._check_meta(meta)

        if self.expression_norm_scope == "base":
            expr_ref_np = meta[self.base_gene_set].values.astype(np.float32)
            if not np.isfinite(expr_ref_np).all():
                raise ValueError(f"Expression matrix for {slide_name} contains NaN or Inf before normalization.")
            if np.any(expr_ref_np < 0):
                warnings.warn(
                    f"Expression matrix for {slide_name} contains negative values. "
                    "library_size_normalize + log transform may be inappropriate."
                )
            exp_ref_info = scp.transform.log(
                scp.normalize.library_size_normalize(expr_ref_np)
            ).astype(np.float32)
            exp_ref_df = pd.DataFrame(exp_ref_info, index=meta.index, columns=self.base_gene_set)
            exp_info = exp_ref_df[self.gene_set].values.astype(np.float32)
        else:
            expr_np = meta[self.gene_set].values.astype(np.float32)
            if not np.isfinite(expr_np).all():
                raise ValueError(f"Expression matrix for {slide_name} contains NaN or Inf before normalization.")
            if np.any(expr_np < 0):
                warnings.warn(
                    f"Expression matrix for {slide_name} contains negative values. "
                    "library_size_normalize + log transform may be inappropriate."
                )
            exp_info = scp.transform.log(
                scp.normalize.library_size_normalize(expr_np)
            ).astype(np.float32)

        if not np.isfinite(exp_info).all():
            raise ValueError(f"Expression matrix for {slide_name} contains NaN or Inf after normalization.")

        center_info = np.floor(
            meta[["pxl_col_in_fullres", "pxl_row_in_fullres"]].values
        ).astype(np.int32)

        self.exp = torch.from_numpy(exp_info)              # (N, G)
        self.pixel_xy = torch.from_numpy(center_info)      # (N, 2), x/y

        self.he = self._open_he_array(he_tif_path)
        self._he_layout = self._infer_layout(self.he)
        self._he_hw = self._get_hw(self.he, self._he_layout)

        # Keep augmentation modest and morphology-safe.
        self.color_jitter = T.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.02,
        )

        if self.check_dataset:
            self._dataset_sanity_report()

    def _check_basic_files(self):
        cnt_path = os.path.join(self.base_dir, self.slide_name, self.expression_file)
        pos_path = os.path.join(self.base_dir, self.slide_name, "subposition.parquet")
        check_file(cnt_path, f"{self.slide_name}/{self.expression_file}")
        check_file(pos_path, f"{self.slide_name}/subposition.parquet")
        check_file(self.he_tif_path, "he_tif_path")
        check_file(self.hvg_file, "hvg_file")

    def _open_he_array(self, he_tif_path: str):
        try:
            return tifffile.memmap(he_tif_path)
        except ValueError:
            import zarr
            with tifffile.TiffFile(he_tif_path) as tif:
                z = tif.series[0].aszarr()
            return zarr.open(z, mode="r")

    def _infer_layout(self, he) -> str:
        if getattr(he, "ndim", None) != 3:
            raise ValueError(f"HE must be 3D, got ndim={getattr(he, 'ndim', None)}")

        s0, s1, s2 = he.shape
        if s0 in (3, 4) and s2 not in (3, 4):
            return "CHW"
        if s2 in (3, 4) and s0 not in (3, 4):
            return "HWC"
        return "HWC"

    @staticmethod
    def _get_hw(he, layout: str) -> Tuple[int, int]:
        if layout == "HWC":
            H, W, _ = he.shape
        else:
            _, H, W = he.shape
        return int(H), int(W)

    def __len__(self) -> int:
        return int(self.exp.shape[0])

    def _read_patch_hwc_uint8(self, x: int, y: int, r: int) -> np.ndarray:
        """
        Read a centered (2r, 2r, 3) RGB patch.
        Correctly handles borders and HWC/CHW TIFF layout.
        """
        he = self.he

        if self._he_layout == "HWC":
            H, W, C = he.shape
            y0, y1 = y - r, y + r
            x0, x1 = x - r, x + r
            sy0, sy1 = max(0, y0), min(H, y1)
            sx0, sx1 = max(0, x0), min(W, x1)

            patch = np.zeros((2 * r, 2 * r, 3), dtype=np.uint8)
            if sy1 > sy0 and sx1 > sx0:
                src = he[sy0:sy1, sx0:sx1, :]
                if src.shape[-1] >= 3:
                    src = src[..., :3]
                dy0 = sy0 - y0
                dx0 = sx0 - x0
                patch[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = np.asarray(src, dtype=np.uint8)
            return patch

        C, H, W = he.shape
        y0, y1 = y - r, y + r
        x0, x1 = x - r, x + r
        sy0, sy1 = max(0, y0), min(H, y1)
        sx0, sx1 = max(0, x0), min(W, x1)

        patch = np.zeros((2 * r, 2 * r, 3), dtype=np.uint8)
        if sy1 > sy0 and sx1 > sx0:
            src = he[:3, sy0:sy1, sx0:sx1]
            src = np.moveaxis(np.asarray(src, dtype=np.uint8), 0, -1)
            dy0 = sy0 - y0
            dx0 = sx0 - x0
            patch[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = src
        return patch

    def _augment_pil(self, img: Image.Image) -> Image.Image:
        if random.random() < 0.5:
            img = TF.hflip(img)
        if random.random() < 0.5:
            img = TF.vflip(img)
        if random.random() < 0.5:
            angle = random.choice([0, 90, 180, 270])
            if angle != 0:
                img = TF.rotate(img, angle)
        if random.random() < 0.8:
            img = self.color_jitter(img)
        return img

    def __getitem__(self, idx: int):
        x, y = self.pixel_xy[idx].tolist()
        patch = self._read_patch_hwc_uint8(x=x, y=y, r=self.r)

        img = Image.fromarray(patch)
        if self.train and self.train_aug:
            img = self._augment_pil(img)

        patch = TF.to_tensor(img)
        if patch.shape[-2:] != (224, 224):
            patch = TF.resize(patch, [224, 224])

        if self.normal:
            patch = TF.normalize(
                patch,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )

        return patch, self.pixel_xy[idx].to(torch.int32), self.exp[idx]

    # ----- meta loaders -----
    def get_cnt(self, name: str) -> pd.DataFrame:
        path = os.path.join(self.base_dir, name, self.expression_file)
        df = pd.read_csv(path, sep=",")
        if df.shape[1] < 2:
            raise ValueError(f"{path} has too few columns.")
        barcode_col = df.columns[0]
        if df[barcode_col].duplicated().any():
            raise ValueError(f"{path} contains duplicated barcode/index values in the first column.")
        df = df.set_index(barcode_col)
        return df

    def get_pos(self, name: str) -> pd.DataFrame:
        path = os.path.join(self.base_dir, name, "subposition.parquet")
        pos = pd.read_parquet(path)
        missing_cols = [c for c in self.REQUIRED_POS_COLS if c not in pos.columns]
        if missing_cols:
            raise ValueError(f"{path} is missing required columns: {missing_cols}")
        if pos["barcode"].duplicated().any():
            raise ValueError(f"{path} contains duplicated barcodes.")
        return pos

    def get_meta(self, name: str) -> pd.DataFrame:
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join(pos.set_index("barcode"), how="left")
        if meta[["pxl_col_in_fullres", "pxl_row_in_fullres"]].isna().any().any():
            missing_n = int(meta[["pxl_col_in_fullres", "pxl_row_in_fullres"]].isna().any(axis=1).sum())
            raise ValueError(
                f"{missing_n} barcodes in {name}/{self.expression_file} cannot be matched to subposition.parquet."
            )
        return meta

    def _check_meta(self, meta: pd.DataFrame):
        required_gene_set = getattr(self, "_required_gene_set", self.gene_set)
        missing_genes = [g for g in required_gene_set if g not in meta.columns]
        if len(missing_genes) > 0:
            raise ValueError(
                f"{len(missing_genes)} required genes are missing from {self.slide_name}/{self.expression_file}. "
                f"Examples: {missing_genes[:10]}"
            )

        coord_cols = ["pxl_col_in_fullres", "pxl_row_in_fullres"]
        if meta[coord_cols].isna().any().any():
            raise ValueError(f"Coordinates contain NaN in slide={self.slide_name}.")
        if not np.isfinite(meta[coord_cols].values.astype(np.float64)).all():
            raise ValueError(f"Coordinates contain Inf in slide={self.slide_name}.")

    def _dataset_sanity_report(self):
        H, W = self._he_hw
        xy = self.pixel_xy.numpy()
        x = xy[:, 0]
        y = xy[:, 1]
        outside = (x < 0) | (x >= W) | (y < 0) | (y >= H)
        border_crop = (x - self.r < 0) | (x + self.r > W) | (y - self.r < 0) | (y + self.r > H)

        print("-" * 90)
        print(f"[DATASET CHECK] slide={self.slide_name}")
        print(f"  base_dir:      {self.base_dir}")
        print(f"  he_tif_path:   {self.he_tif_path}")
        print(f"  HE shape:      {tuple(self.he.shape)}, layout={self._he_layout}, H={H}, W={W}")
        print(f"  spots/bins:    {len(self):,}")
        print(f"  genes:         {len(self.gene_set):,}")
        print(f"  gene_mode:     {self.gene_mode}")
        print(f"  gene_num:      {self.gene_num}")
        print(f"  markers:       {self.marker_gene_set}")
        print(f"  norm_scope:    {self.expression_norm_scope}")
        print(f"  exp shape:     {tuple(self.exp.shape)}")
        print(f"  x range:       [{int(x.min())}, {int(x.max())}]")
        print(f"  y range:       [{int(y.min())}, {int(y.max())}]")
        print(f"  outside image: {int(outside.sum()):,}")
        print(f"  border crops:  {int(border_crop.sum()):,}")
        print(f"  train:         {self.train}")
        print(f"  train_aug:     {self.train_aug}")
        print(f"  normal:        {self.normal}")
        if int(outside.sum()) > 0:
            raise ValueError(
                f"Found {int(outside.sum())} coordinates outside the HE image for slide={self.slide_name}."
            )
        print("-" * 90)


# =============================================================================
# iBOT-MambaVision encoder and two-layer Linear head
# =============================================================================
class IBOTMambaEncoder(nn.Module):
    """
    iBOT-MambaVision encoder wrapper.

    Released feature aggregation module: HFAM.
    """

    VALID_POOLS = ["HFAM"]

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        arch: str = "mamba_vision_base",
        strict: bool = True,
        return_all_tokens: bool = False,
        masked_im_modeling: bool = False,
        feature_pool: str = "HFAM",
    ):
        super().__init__()

        try:
            from spavista import models
        except Exception as e:
            raise ImportError(
                "Cannot import `models`. Run this script from the project root or set --project_root."
            ) from e

        if not hasattr(models, arch):
            raise ValueError(
                f"Unknown MambaVision arch: {arch}. "
                f"Please check models/__init__.py and models/mambavision_ibot.py."
            )

        if feature_pool not in self.VALID_POOLS:
            raise ValueError(
                f"Unknown feature_pool={feature_pool}. Choose from: {self.VALID_POOLS}."
            )

        self.arch = arch
        self.feature_pool = feature_pool

        self.encoder = getattr(models, arch)(
            pretrained=False,
            return_all_tokens=return_all_tokens,
            masked_im_modeling=masked_im_modeling,
        )

        if hasattr(self.encoder, "embed_dim"):
            self.encoder_dim = self.encoder.embed_dim
        elif hasattr(self.encoder, "num_features"):
            self.encoder_dim = self.encoder.num_features
        else:
            raise AttributeError(
                f"Cannot infer output dim from encoder arch={arch}. "
                "Expected encoder.embed_dim or encoder.num_features."
            )

        self.output_dim = self.encoder_dim * 2

        if checkpoint_path is not None and str(checkpoint_path).strip() != "":
            self.load_ibot_mamba_weights(checkpoint_path, strict=strict)
        else:
            warnings.warn("No encoder checkpoint is provided. A checkpoint is required for downstream evaluation.")

    @staticmethod
    def _select_state_dict(ckpt):
        if isinstance(ckpt, dict):
            for key in ["state_dict", "model", "student", "teacher", "backbone", "encoder"]:
                if key in ckpt and isinstance(ckpt[key], dict):
                    return ckpt[key]
            return ckpt
        raise ValueError("Unsupported checkpoint format. Expected a dict or a dict containing state_dict/model/student/teacher.")

    @staticmethod
    def _strip_common_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        prefixes = [
            "module.backbone.",
            "backbone.",
            "module.encoder.",
            "encoder.",
            "module.",
            "ibot_mamba.encoder.",
            "ibot_mamba.",
        ]
        new_state_dict = {}
        for k, v in state_dict.items():
            nk = k
            changed = True
            while changed:
                changed = False
                for p in prefixes:
                    if nk.startswith(p):
                        nk = nk[len(p):]
                        changed = True
                        break
            new_state_dict[nk] = v
        return new_state_dict

    def load_ibot_mamba_weights(self, checkpoint_path: str, strict: bool = True):
        check_file(checkpoint_path, "encoder checkpoint")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = self._select_state_dict(ckpt)
        state_dict = self._strip_common_prefixes(state_dict)

        msg = self.encoder.load_state_dict(state_dict, strict=strict)

        print(f"[IBOTMambaEncoder] arch: {self.arch}")
        print(f"[IBOTMambaEncoder] feature_pool: {self.feature_pool}")
        print(f"[IBOTMambaEncoder] encoder_dim: {self.encoder_dim}")
        print(f"[IBOTMambaEncoder] output_dim: {self.output_dim}")
        print(f"[IBOTMambaEncoder] Loaded checkpoint: {checkpoint_path}")
        print(f"[IBOTMambaEncoder] strict_load: {strict}")
        print(f"[IBOTMambaEncoder] missing keys: {msg.missing_keys}")
        print(f"[IBOTMambaEncoder] unexpected keys: {msg.unexpected_keys}")

    @staticmethod
    def _to_bchw(x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise RuntimeError(f"Expected 4D feature map, got shape={tuple(x.shape)}")

        common_channels = {
            64, 80, 96, 128, 160, 192, 256, 320, 384,
            512, 640, 768, 784, 1024, 1536, 1568, 2048
        }
        if x.shape[1] not in common_channels and x.shape[-1] in common_channels:
            x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def _get_mambavision_model(self):
        if hasattr(self.encoder, "model"):
            model = self.encoder.model
        else:
            model = self.encoder

        if not hasattr(model, "patch_embed") or not hasattr(model, "levels"):
            raise RuntimeError(
                "Cannot find model.patch_embed or model.levels. "
                "Please check the MambaVision wrapper structure."
            )
        return model

    def _forward_mambavision_all_levels(self, x: torch.Tensor) -> List[torch.Tensor]:
        model = self._get_mambavision_model()
        x = model.patch_embed(x)

        outs = []
        for level in model.levels:
            x = level(x)
            if isinstance(x, (tuple, list)):
                x = x[0]
            outs.append(self._to_bchw(x))
        return outs

    def _forward_late_stage_maps(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        model = self._get_mambavision_model()
        outs = self._forward_mambavision_all_levels(x)

        if len(outs) < 4:
            raise RuntimeError(f"Expected 4 MambaVision levels, got {len(outs)}")

        feat_a = outs[2]
        feat_b = outs[3]

        if hasattr(model, "norm") and model.norm is not None:
            try:
                feat_b = model.norm(feat_b)
            except Exception:
                pass

        feat_a = self._to_bchw(feat_a)
        feat_b = self._to_bchw(feat_b)
        return feat_a, feat_b

    @staticmethod
    def _gap(feat: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(feat, output_size=1).flatten(1)

    @staticmethod
    def _branch_l2(feat: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        return F.normalize(feat, p=2, dim=1, eps=eps)

    @staticmethod
    def _signed_sqrt(feat: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        return torch.sign(feat) * torch.sqrt(torch.clamp(torch.abs(feat), min=eps))

    @staticmethod
    def _crow_pool(feat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        if feat.dim() != 4:
            raise RuntimeError(f"Expected feat [B,C,H,W], got {tuple(feat.shape)}")

        B, C, H, W = feat.shape
        feat_abs = torch.abs(feat)

        spatial = feat_abs.sum(dim=1, keepdim=True)
        spatial = spatial / (spatial.mean(dim=(2, 3), keepdim=True) + eps)

        ch_mean = feat_abs.mean(dim=(2, 3), keepdim=True)
        active = (feat_abs > ch_mean).float().sum(dim=(2, 3), keepdim=True)

        num_pos = float(H * W)
        channel = torch.log((num_pos + eps) / (active + eps))
        channel = channel / (channel.mean(dim=1, keepdim=True) + eps)

        weighted = feat * spatial * channel
        pooled = weighted.mean(dim=(2, 3))
        return pooled

    def _power_l2_readout_from_maps(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        fa = self._gap(feat_a)
        fb = self._gap(feat_b)
        fa = self._branch_l2(self._signed_sqrt(fa))
        fb = self._branch_l2(self._signed_sqrt(fb))
        return torch.cat([fa, fb], dim=1)

    def _crow_readout_from_maps(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        fa = self._branch_l2(self._crow_pool(feat_a))
        fb = self._branch_l2(self._crow_pool(feat_b))
        return torch.cat([fa, fb], dim=1)

    def _late_stage_readout(self, x):
        """HFAM: average power-normalized and CroW late-stage features."""
        feat_a, feat_b = self._forward_late_stage_maps(x)
        f_power = self._power_l2_readout_from_maps(feat_a, feat_b) if hasattr(self, "_power_l2_readout_from_maps") else self._power_l2_readout(feat_a, feat_b)
        f_crow = self._crow_readout_from_maps(feat_a, feat_b) if hasattr(self, "_crow_readout_from_maps") else self._crow_readout(feat_a, feat_b)
        return 0.5 * f_power + 0.5 * f_crow

    def forward(self, x):
        return self._late_stage_readout(x)


class VisiumHDLinearHeadModel(nn.Module):
    """
    Frozen iBOT-MambaVision encoder + two-layer Linear gene head.

    gene_head:
        encoder_output_dim -> hidden_dim(1024 by default) -> n_genes
    """

    def __init__(
        self,
        n_genes: int,
        encoder_ckpt: str,
        arch: str = "mamba_vision_base",
        feature_pool: str = "HFAM",
        head_hidden_dim: int = 1024,
        strict_load: bool = True,
    ):
        super().__init__()
        self.n_genes = int(n_genes)
        self.arch = arch
        self.feature_pool = feature_pool
        self.head_hidden_dim = int(head_hidden_dim)

        self.ibot_mamba = IBOTMambaEncoder(
            checkpoint_path=encoder_ckpt,
            arch=arch,
            strict=strict_load,
            return_all_tokens=False,
            masked_im_modeling=False,
            feature_pool=feature_pool,
        )

        self.encoder_dim = int(self.ibot_mamba.output_dim)
        for p in self.ibot_mamba.parameters():
            p.requires_grad = False
        self.ibot_mamba.eval()

        self.gene_head = nn.Sequential(
            nn.Linear(self.encoder_dim, self.head_hidden_dim),
            nn.Linear(self.head_hidden_dim, self.n_genes),
        )

    def forward(self, patches: torch.Tensor, centers: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.ibot_mamba.eval()
        with torch.no_grad():
            h = self.ibot_mamba(patches)
        pred = self.gene_head(h)
        return pred


def build_model(args, n_genes: int) -> VisiumHDLinearHeadModel:
    model = VisiumHDLinearHeadModel(
        n_genes=n_genes,
        encoder_ckpt=args.encoder_ckpt,
        arch=args.arch,
        feature_pool="HFAM",
        head_hidden_dim=args.head_hidden_dim,
        strict_load=args.strict_load,
    )
    return model


# =============================================================================
# Linear-head checkpoint utils
# =============================================================================
def get_linear_head_ckpt_path(args) -> str:
    if args.linear_head_ckpt and str(args.linear_head_ckpt).strip() != "":
        return args.linear_head_ckpt
    return os.path.join(args.output_dir, "linear_head_ckpt", "linear_head_last.pt")


def save_linear_head_checkpoint(model: VisiumHDLinearHeadModel, args, gene_names: List[str]) -> str:
    ckpt_path = get_linear_head_ckpt_path(args)
    ensure_dir(os.path.dirname(ckpt_path))

    payload = {
        "checkpoint_type": "visiumhd_linear_head_only",
        "saved_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args).copy(),
        "model_config": {
            "n_genes": int(model.n_genes),
            "gene_names": list(gene_names),
            "arch": model.arch,
            "feature_pool": model.feature_pool,
            "encoder_ckpt": args.encoder_ckpt,
            "encoder_dim": int(model.encoder_dim),
            "head_hidden_dim": int(model.head_hidden_dim),
            "freeze_encoder": True,
            "head_structure": f"Linear({model.encoder_dim}, {model.head_hidden_dim}) -> Linear({model.head_hidden_dim}, {model.n_genes})",
        },
        "gene_head_state_dict": {
            k: v.detach().cpu() for k, v in model.gene_head.state_dict().items()
        },
    }

    tmp_path = ckpt_path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, ckpt_path)

    if not os.path.exists(ckpt_path) or os.path.getsize(ckpt_path) == 0:
        raise RuntimeError(f"Failed to save non-empty linear head checkpoint: {ckpt_path}")

    print(f"[INFO] Saved linear head checkpoint to: {ckpt_path}")
    print(f"[INFO] Linear head checkpoint size: {os.path.getsize(ckpt_path) / 1024:.2f} KB")
    return ckpt_path


def load_linear_head_checkpoint(model: VisiumHDLinearHeadModel, args) -> VisiumHDLinearHeadModel:
    ckpt_path = get_linear_head_ckpt_path(args)
    check_file(ckpt_path, "linear_head_ckpt")

    payload = torch.load(ckpt_path, map_location="cpu")
    if payload.get("checkpoint_type") != "visiumhd_linear_head_only":
        raise ValueError(f"Unsupported checkpoint type in {ckpt_path}: {payload.get('checkpoint_type')}")

    model.gene_head.load_state_dict(payload["gene_head_state_dict"], strict=True)
    print(f"[INFO] Loaded linear head checkpoint from: {ckpt_path}")
    return model


# =============================================================================
# DataLoader helpers
# =============================================================================


# =============================================================================
# Training
# =============================================================================


# =============================================================================
# Metrics
# =============================================================================
def init_gene_metric_stats(num_genes: int) -> Dict[str, np.ndarray]:
    return {
        "n": 0,
        "sum_y": np.zeros(num_genes, dtype=np.float64),
        "sum_p": np.zeros(num_genes, dtype=np.float64),
        "sum_y2": np.zeros(num_genes, dtype=np.float64),
        "sum_p2": np.zeros(num_genes, dtype=np.float64),
        "sum_yp": np.zeros(num_genes, dtype=np.float64),
        "sum_abs": np.zeros(num_genes, dtype=np.float64),
        "sum_se": np.zeros(num_genes, dtype=np.float64),
    }


def update_gene_metric_stats(stats: Dict[str, np.ndarray], y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    stats["n"] += y_true.shape[0]
    stats["sum_y"] += y_true.sum(axis=0)
    stats["sum_p"] += y_pred.sum(axis=0)
    stats["sum_y2"] += (y_true ** 2).sum(axis=0)
    stats["sum_p2"] += (y_pred ** 2).sum(axis=0)
    stats["sum_yp"] += (y_true * y_pred).sum(axis=0)
    diff = y_pred - y_true
    stats["sum_abs"] += np.abs(diff).sum(axis=0)
    stats["sum_se"] += (diff ** 2).sum(axis=0)


def finalize_gene_metric_stats(stats: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = float(stats["n"])
    cov = n * stats["sum_yp"] - stats["sum_y"] * stats["sum_p"]
    var_y = n * stats["sum_y2"] - stats["sum_y"] ** 2
    var_p = n * stats["sum_p2"] - stats["sum_p"] ** 2
    denom = np.sqrt(np.maximum(var_y, 0.0) * np.maximum(var_p, 0.0))
    pcc = np.divide(cov, denom, out=np.full_like(cov, np.nan), where=denom > 1e-12)
    mse = stats["sum_se"] / max(n, 1.0)
    mae = stats["sum_abs"] / max(n, 1.0)
    return pcc.astype(np.float32), mse.astype(np.float32), mae.astype(np.float32)


def batch_cell_pcc(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    yt = y_true - y_true.mean(axis=1, keepdims=True)
    yp = y_pred - y_pred.mean(axis=1, keepdims=True)
    denom = np.sqrt((yt ** 2).sum(axis=1) * (yp ** 2).sum(axis=1))
    corr = np.divide((yt * yp).sum(axis=1), denom, out=np.full(y_true.shape[0], np.nan), where=denom > 1e-12)
    return corr.astype(np.float32)


def compute_spearman_per_gene(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    if spearmanr is None:
        warnings.warn("scipy is unavailable, SCC will be skipped.")
        return np.full(y_true.shape[1], np.nan, dtype=np.float32)

    g = y_true.shape[1]
    scc = np.full(g, np.nan, dtype=np.float32)
    for j in tqdm(range(g), desc="Compute SCC per gene"):
        yt = np.asarray(y_true[:, j])
        yp = np.asarray(y_pred[:, j])
        if np.std(yt) < 1e-12 or np.std(yp) < 1e-12:
            continue
        val = spearmanr(yt, yp).correlation
        scc[j] = np.float32(val) if val is not None else np.nan
    return scc


@torch.no_grad()
def evaluate_model(
    model: VisiumHDLinearHeadModel,
    test_loader: DataLoader,
    test_ds: ViT_VisiumHD,
    args,
    device: torch.device,
) -> Dict[str, float]:
    model.to(device)
    model.eval()
    model.ibot_mamba.eval()

    num_test = len(test_ds)
    gene_names = test_ds.gene_set
    num_genes = len(gene_names)

    stats = init_gene_metric_stats(num_genes)
    cell_corr_all = []

    need_pred_file = args.save_pred or (not args.no_scc)
    pred_path = os.path.join(args.output_dir, "pred_test_8um.npy")
    tmp_pred_path = os.path.join(args.output_dir, "tmp_pred_for_scc.npy")
    active_pred_path = pred_path if args.save_pred else tmp_pred_path
    pred_memmap = None
    write_start = 0

    if need_pred_file:
        pred_memmap = np.lib.format.open_memmap(
            active_pred_path,
            mode="w+",
            dtype=np.float32,
            shape=(num_test, num_genes),
        )
        print(f"[INFO] Prediction memmap created: {active_pred_path}")

    pbar = tqdm(test_loader, desc="Evaluate test")
    for batch in pbar:
        patch, center, exp = batch
        patch = patch.to(device, non_blocking=True)
        center = center.to(device, non_blocking=True)
        exp_np = exp.numpy().astype(np.float32)

        with autocast_context(args, device):
            pred = model(patch, center)
        pred_np = pred.detach().float().cpu().numpy().astype(np.float32)

        update_gene_metric_stats(stats, exp_np, pred_np)
        cell_corr_all.append(batch_cell_pcc(exp_np, pred_np))

        if pred_memmap is not None:
            bsz = pred_np.shape[0]
            pred_memmap[write_start:write_start + bsz] = pred_np
            write_start += bsz

    if pred_memmap is not None:
        pred_memmap.flush()
        del pred_memmap

    pcc, mse, mae = finalize_gene_metric_stats(stats)

    if args.no_scc:
        scc = np.full_like(pcc, np.nan, dtype=np.float32)
    else:
        y_true_for_scc = test_ds.exp.numpy().astype(np.float32)
        y_pred_for_scc = np.load(active_pred_path, mmap_mode="r")
        scc = compute_spearman_per_gene(y_true_for_scc, y_pred_for_scc)

    cell_corr = np.concatenate(cell_corr_all, axis=0) if len(cell_corr_all) else np.array([], dtype=np.float32)
    valid_cell_corr = cell_corr[~np.isnan(cell_corr)]

    # HEG/HVG top-50 based on the true 8um test expression.
    y_test = test_ds.exp.numpy().astype(np.float32)
    ind_heg50 = np.argsort(np.sum(y_test, axis=0))[-50:]
    ind_hvg50 = np.argsort(np.var(y_test, axis=0))[-50:]
    heg50_pcc = pcc[ind_heg50]
    hvg50_pcc = pcc[ind_hvg50]
    heg50_pcc = heg50_pcc[~np.isnan(heg50_pcc)]
    hvg50_pcc = hvg50_pcc[~np.isnan(hvg50_pcc)]

    per_gene = pd.DataFrame({
        "gene": gene_names,
        "PCC": pcc,
        "SCC": scc,
        "MSE": mse,
        "MAE": mae,
    })
    per_gene_path = os.path.join(args.output_dir, "per_gene_metrics_8um.csv")
    per_gene.to_csv(per_gene_path, index=False)

    summary = {
        "train_slide": args.train_slide,
        "test_slide": args.test_slide,
        "train_aug": False,
        "normal": bool(args.normal),
        "train_spots": int(args._train_spots),
        "test_spots": int(num_test),
        "num_genes": int(num_genes),
        "gene_mode": str(getattr(args, "gene_mode", "all")),
        "gene_num": int(getattr(args, "gene_num", 0)),
        "marker_genes": str(getattr(args, "marker_genes", "")),
        "marker_gene_file": str(getattr(args, "marker_gene_file", "")),
        "marker_from_hvg_tail": int(getattr(args, "marker_from_hvg_tail", 0)),
        "expression_norm_scope": str(getattr(args, "expression_norm_scope", "base")),
        "arch": args.arch,
        "feature_pool": "HFAM",
        "encoder_ckpt": args.encoder_ckpt,
        "head_structure": f"Linear({model.encoder_dim}, {model.head_hidden_dim}) -> Linear({model.head_hidden_dim}, {model.n_genes})",
        "mean_gene_PCC": float(np.nanmean(pcc)),
        "median_gene_PCC": float(np.nanmedian(pcc)),
        "max_gene_PCC": float(np.nanmax(pcc)),
        "median_HEG50_gene_PCC": float(np.nanmedian(heg50_pcc)) if len(heg50_pcc) else float("nan"),
        "median_HVG50_gene_PCC": float(np.nanmedian(hvg50_pcc)) if len(hvg50_pcc) else float("nan"),
        "mean_gene_SCC": float(np.nanmean(scc)),
        "median_gene_SCC": float(np.nanmedian(scc)),
        "mean_gene_MSE": float(np.nanmean(mse)),
        "median_gene_MSE": float(np.nanmedian(mse)),
        "mean_gene_MAE": float(np.nanmean(mae)),
        "median_gene_MAE": float(np.nanmedian(mae)),
        "median_cell_PCC": float(np.nanmedian(valid_cell_corr)) if len(valid_cell_corr) else float("nan"),
        "mean_cell_PCC": float(np.nanmean(valid_cell_corr)) if len(valid_cell_corr) else float("nan"),
    }

    summary_path = os.path.join(args.output_dir, "summary_metrics_8um.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    cell_path = os.path.join(args.output_dir, "cell_pcc_8um.npy")
    np.save(cell_path, cell_corr.astype(np.float32))

    if (not args.save_pred) and os.path.exists(tmp_pred_path):
        try:
            os.remove(tmp_pred_path)
            print(f"[INFO] Removed temporary prediction file: {tmp_pred_path}")
        except Exception as e:
            warnings.warn(f"Failed to remove temporary prediction file {tmp_pred_path}: {e}")

    print("\n" + "=" * 90)
    print("Final 8um test performance")
    print(f"Median cell PCC:      {summary['median_cell_PCC']:.6f}")
    print(f"Mean gene PCC:        {summary['mean_gene_PCC']:.6f}")
    print(f"Median gene PCC:      {summary['median_gene_PCC']:.6f}")
    print(f"Max gene PCC:         {summary['max_gene_PCC']:.6f}")
    print(f"Median HEG50 PCC:     {summary['median_HEG50_gene_PCC']:.6f}")
    print(f"Median HVG50 PCC:     {summary['median_HVG50_gene_PCC']:.6f}")
    print(f"Mean gene SCC:        {summary['mean_gene_SCC']:.6f}")
    print(f"Median gene SCC:      {summary['median_gene_SCC']:.6f}")
    print(f"Mean gene MSE:        {summary['mean_gene_MSE']:.6f}")
    print(f"Median gene MSE:      {summary['median_gene_MSE']:.6f}")
    print(f"Mean gene MAE:        {summary['mean_gene_MAE']:.6f}")
    print(f"Median gene MAE:      {summary['median_gene_MAE']:.6f}")
    print("=" * 90)
    print(f"[SAVED] {per_gene_path}")
    print(f"[SAVED] {summary_path}")
    print(f"[SAVED] {cell_path}")
    if args.save_pred:
        print(f"[SAVED] {pred_path}")

    return summary



# =============================================================================
# Cached feature dataset and feature extraction
# =============================================================================
class CachedFeatureDataset(Dataset):
    """
    Dataset for cached frozen-encoder features.

    Returns:
        feature: (D,), float32
        exp:     (G,), float32
    """

    def __init__(
        self,
        feature_path: str,
        exp_tensor: torch.Tensor,
        load_to_ram: bool = False,
    ):
        super().__init__()
        check_file(feature_path, "cached feature_path")
        self.feature_path = feature_path
        self.load_to_ram = bool(load_to_ram)

        if self.load_to_ram:
            # Keep the on-disk dtype in RAM. The training/eval loops cast to float32
            # before feeding the linear head. This saves RAM for float16 cached features.
            self.features = np.load(feature_path)
        else:
            self.features = np.load(feature_path, mmap_mode="r")

        if self.features.ndim != 2:
            raise ValueError(f"Cached features should be 2D [N,D], got shape={self.features.shape}")

        self.exp = exp_tensor.detach().cpu().float()
        if len(self.features) != self.exp.shape[0]:
            raise ValueError(
                f"Cached feature N={len(self.features)} does not match expression N={self.exp.shape[0]} "
                f"for feature_path={feature_path}"
            )

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, idx: int):
        # np.memmap slices can be non-writable; copy avoids PyTorch warnings and unsafe shared writes.
        # Keep cached dtype here, usually float16. The training/eval loop casts to float32.
        feat = np.asarray(self.features[idx])
        if not self.load_to_ram:
            feat = feat.copy()
        feat = torch.from_numpy(feat)
        exp = self.exp[idx]
        return feat, exp


def make_feature_training_loader(dataset: CachedFeatureDataset, args, train: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    return DataLoader(
        dataset,
        batch_size=args.head_batch_size,
        shuffle=train,
        num_workers=args.feature_num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=args.feature_num_workers > 0,
    )


def make_feature_extract_loader(dataset: Dataset, args) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    return DataLoader(
        dataset,
        batch_size=args.feature_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=args.num_workers > 0,
    )


def sanitize_for_filename(x: str) -> str:
    x = str(x)
    for ch in ["/", "\\", ":", " ", "\t", "\n", ",", ";"]:
        x = x.replace(ch, "_")
    return x


def default_feature_paths(args) -> Tuple[str, str]:
    if args.feature_dir is None or str(args.feature_dir).strip() == "":
        feature_dir = os.path.join(args.output_dir, "cached_features")
    else:
        feature_dir = args.feature_dir
    ensure_dir(feature_dir)

    suffix = (
        f"arch-{sanitize_for_filename(args.arch)}_"
        "pool-HFAM_"
        f"r{int(args.crop_radius)}_"
        f"normal{int(bool(args.normal))}_"
        f"dtype-{sanitize_for_filename(getattr(args, 'feature_cache_dtype', 'float16'))}_"
        f"dimauto"
    )

    train_path = args.train_feature_path
    test_path = args.test_feature_path

    def existing_cache(slide, split):
        if not args.feature_dir or args.force_extract_features:
            return ""
        prefix = f"features_{sanitize_for_filename(slide)}_{split}_"
        candidates = [
            os.path.join(feature_dir, name) for name in os.listdir(feature_dir)
            if name.startswith(prefix) and name.endswith(".npy")
        ]
        compatible = []
        for path in candidates:
            meta_path = path + ".meta.json"
            if not os.path.isfile(meta_path):
                continue
            try:
                with open(meta_path, "r", encoding="utf-8") as handle:
                    meta = json.load(handle)
                pool = meta.get("feature_pool")
                if (
                    meta.get("arch") == args.arch
                    and pool in ("HFAM", "stage3_stage4_power_crow_avg")
                    and int(meta.get("crop_radius", -1)) == int(args.crop_radius)
                    and bool(meta.get("normal")) == bool(args.normal)
                ):
                    compatible.append(path)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        if len(compatible) > 1:
            raise ValueError(f"Multiple compatible {split} feature caches found; set --{split}_feature_path explicitly: {compatible}")
        if compatible:
            print(f"[INFO] Discovered compatible {split} feature cache: {compatible[0]}")
            return compatible[0]
        return ""

    if train_path is None or str(train_path).strip() == "":
        train_path = existing_cache(args.train_slide, "train")
    if train_path is None or str(train_path).strip() == "":
        train_path = os.path.join(
            feature_dir,
            f"features_{sanitize_for_filename(args.train_slide)}_train_{suffix}.npy",
        )
    if test_path is None or str(test_path).strip() == "":
        test_path = existing_cache(args.test_slide, "test")
    if test_path is None or str(test_path).strip() == "":
        test_path = os.path.join(
            feature_dir,
            f"features_{sanitize_for_filename(args.test_slide)}_test_{suffix}.npy",
        )

    ensure_dir(os.path.dirname(train_path))
    ensure_dir(os.path.dirname(test_path))
    return train_path, test_path


def save_feature_metadata(
    feature_path: str,
    args,
    split_name: str,
    dataset: ViT_VisiumHD,
    model: VisiumHDLinearHeadModel,
    feature_shape: Tuple[int, int],
):
    meta = {
        "split_name": split_name,
        "feature_path": feature_path,
        "saved_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "slide_name": dataset.slide_name,
        "base_dir": dataset.base_dir,
        "he_tif_path": dataset.he_tif_path,
        "hvg_file": dataset.hvg_file,
        "num_bins": int(len(dataset)),
        "num_genes": int(len(dataset.gene_set)),
        "gene_names": list(dataset.gene_set),
        "gene_mode": getattr(dataset, "gene_mode", "all"),
        "gene_num": int(getattr(dataset, "gene_num", 0)),
        "marker_gene_set": list(getattr(dataset, "marker_gene_set", [])),
        "expression_norm_scope": getattr(dataset, "expression_norm_scope", "base"),
        "base_gene_num": int(len(getattr(dataset, "base_gene_set", dataset.gene_set))),
        "feature_shape": [int(feature_shape[0]), int(feature_shape[1])],
        "feature_dtype": str(np.dtype(resolve_feature_cache_dtype(args))),
        "arch": args.arch,
        "feature_pool": "HFAM",
        "encoder_ckpt": args.encoder_ckpt,
        "encoder_dim": int(model.encoder_dim),
        "normal": bool(args.normal),
        "crop_radius": int(args.crop_radius),
        "train_aug": False,
        "note": "Frozen encoder features were extracted deterministically with train_aug=False.",
    }
    meta_path = feature_path + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)
    print(f"[SAVED] {meta_path}")


def validate_cached_features(
    feature_path: str,
    expected_n: int,
    expected_dim: int,
    split_name: str,
    args=None,
):
    check_file(feature_path, f"{split_name} cached features")
    arr = np.load(feature_path, mmap_mode="r")
    if arr.ndim != 2:
        raise ValueError(f"{split_name} cached features should be 2D [N,D], got shape={arr.shape}")
    if int(arr.shape[0]) != int(expected_n):
        raise ValueError(
            f"{split_name} cached features have N={arr.shape[0]}, expected N={expected_n}. "
            f"Path: {feature_path}"
        )
    if int(arr.shape[1]) != int(expected_dim):
        raise ValueError(
            f"{split_name} cached features have D={arr.shape[1]}, expected encoder_dim={expected_dim}. "
            f"Path: {feature_path}"
        )
    if args is not None:
        expected_dtype = np.dtype(resolve_feature_cache_dtype(args))
        if np.dtype(arr.dtype) != expected_dtype:
            warnings.warn(
                f"{split_name} cached feature dtype is {arr.dtype}, but current --feature_cache_dtype "
                f"is {expected_dtype}. Reusing it is allowed. To regenerate smaller float16 files, "
                f"set --force_extract_features True or use a new feature path."
            )
    print(f"[INFO] Using existing {split_name} cached features: {feature_path}, shape={arr.shape}, dtype={arr.dtype}")
    return arr.shape


@torch.no_grad()
def extract_features_to_npy(
    model: VisiumHDLinearHeadModel,
    dataset: ViT_VisiumHD,
    loader: DataLoader,
    args,
    device: torch.device,
    out_path: str,
    split_name: str,
):
    """
    Extract frozen encoder features once and save to .npy memmap.

    Important:
        This function assumes dataset was created with train_aug=False.
        This is required for deterministic feature caching.
    """
    if getattr(dataset, "train_aug", False):
        raise ValueError(
            f"{split_name} dataset has train_aug=True. Cached feature extraction requires train_aug=False."
        )

    ensure_dir(os.path.dirname(out_path))

    model.ibot_mamba.to(device)
    model.ibot_mamba.eval()

    num_samples = len(dataset)
    feature_dim = int(model.encoder_dim)

    tmp_path = out_path + ".tmp.npy"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    cache_dtype = resolve_feature_cache_dtype(args)
    print(f"[INFO] Saving {split_name} cached features with dtype={np.dtype(cache_dtype)}")

    feat_memmap = np.lib.format.open_memmap(
        tmp_path,
        mode="w+",
        dtype=cache_dtype,
        shape=(num_samples, feature_dim),
    )

    write_start = 0
    pbar = tqdm(loader, desc=f"Extract {split_name} features")
    for batch in pbar:
        patch, center, exp = batch
        patch = patch.to(device, non_blocking=True)

        with autocast_context(args, device):
            feat = model.ibot_mamba(patch)

        feat_np = feat.detach().float().cpu().numpy().astype(cache_dtype, copy=False)
        bsz = feat_np.shape[0]
        feat_memmap[write_start:write_start + bsz] = feat_np
        write_start += bsz
        pbar.set_postfix(saved=f"{write_start}/{num_samples}")

    if write_start != num_samples:
        raise RuntimeError(f"{split_name} feature extraction wrote {write_start} rows, expected {num_samples}")

    feat_memmap.flush()
    del feat_memmap

    os.replace(tmp_path, out_path)
    print(f"[SAVED] {split_name} features: {out_path}")

    save_feature_metadata(
        feature_path=out_path,
        args=args,
        split_name=split_name,
        dataset=dataset,
        model=model,
        feature_shape=(num_samples, feature_dim),
    )

    model.ibot_mamba.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out_path


def load_or_extract_cached_features(
    model: VisiumHDLinearHeadModel,
    train_ds: ViT_VisiumHD,
    test_ds: ViT_VisiumHD,
    args,
    device: torch.device,
) -> Tuple[str, str]:
    train_feature_path, test_feature_path = default_feature_paths(args)

    need_train_extract = args.force_extract_features or (not os.path.exists(train_feature_path))
    need_test_extract = args.force_extract_features or (not os.path.exists(test_feature_path))

    if need_train_extract or need_test_extract:
        print("[INFO] Cached features are missing or --force_extract_features=True.")
        print("[INFO] Build deterministic image dataloaders and extract frozen encoder features.")
    else:
        print("[INFO] Both train/test cached feature files already exist. Skip encoder forwarding.")

    if need_train_extract:
        train_extract_loader = make_feature_extract_loader(train_ds, args)
        extract_features_to_npy(
            model=model,
            dataset=train_ds,
            loader=train_extract_loader,
            args=args,
            device=device,
            out_path=train_feature_path,
            split_name="train",
        )
    else:
        validate_cached_features(
            feature_path=train_feature_path,
            expected_n=len(train_ds),
            expected_dim=model.encoder_dim,
            split_name="train",
            args=args,
        )

    if need_test_extract:
        test_extract_loader = make_feature_extract_loader(test_ds, args)
        extract_features_to_npy(
            model=model,
            dataset=test_ds,
            loader=test_extract_loader,
            args=args,
            device=device,
            out_path=test_feature_path,
            split_name="test",
        )
    else:
        validate_cached_features(
            feature_path=test_feature_path,
            expected_n=len(test_ds),
            expected_dim=model.encoder_dim,
            split_name="test",
            args=args,
        )

    # Final validation after extraction.
    validate_cached_features(
        feature_path=train_feature_path,
        expected_n=len(train_ds),
        expected_dim=model.encoder_dim,
        split_name="train",
            args=args,
    )
    validate_cached_features(
        feature_path=test_feature_path,
        expected_n=len(test_ds),
        expected_dim=model.encoder_dim,
        split_name="test",
            args=args,
    )

    return train_feature_path, test_feature_path


# =============================================================================
# Feature-head training and evaluation
# =============================================================================
def train_head_from_cached_features(
    model: VisiumHDLinearHeadModel,
    train_loader: DataLoader,
    args,
    device: torch.device,
) -> pd.DataFrame:
    """
    Train only the two-layer gene head on cached feature vectors.
    The frozen image encoder is not used here.

    The test set is not accessed during training.
    """
    model.gene_head.to(device)
    model.gene_head.train()

    optimizer = torch.optim.AdamW(
        model.gene_head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = None
    if args.lr_step_size > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.lr_step_size,
            gamma=args.lr_gamma,
        )

    use_scaler = (device.type == "cuda" and args.amp == "fp16")
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    history = []
    for epoch in range(1, args.max_epoch + 1):
        model.gene_head.train()
        running_loss = 0.0
        seen = 0

        pbar = tqdm(train_loader, desc=f"Train head epoch {epoch}/{args.max_epoch}")
        for features, exp in pbar:
            # Cached features can be float16 on disk; cast to float32 for stable head training.
            features = features.to(device, non_blocking=True).float()
            exp = exp.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)

            with autocast_context(args, device):
                pred = model.gene_head(features)
                loss = F.mse_loss(pred.float().view_as(exp), exp)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if args.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.gene_head.parameters(), args.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.gene_head.parameters(), args.grad_clip_norm)
                optimizer.step()

            bsz = int(exp.shape[0])
            running_loss += float(loss.detach().cpu()) * bsz
            seen += bsz
            pbar.set_postfix(loss=running_loss / max(seen, 1))

        if scheduler is not None:
            scheduler.step()

        epoch_loss = running_loss / max(seen, 1)
        current_lr = float(optimizer.param_groups[0]["lr"])
        row = {"epoch": epoch, "train_loss": epoch_loss, "lr": current_lr}
        print(f"[TRAIN] epoch={epoch:03d}, train_loss={epoch_loss:.6f}, lr={current_lr:.6e}")

        history.append(row)

        # Save/update train loss after every epoch so intermediate runs are not lost.
        hist_df = pd.DataFrame(history)
        hist_path = os.path.join(args.output_dir, "train_loss.csv")
        hist_df.to_csv(hist_path, index=False)

    hist_df = pd.DataFrame(history)
    hist_path = os.path.join(args.output_dir, "train_loss.csv")
    hist_df.to_csv(hist_path, index=False)
    print(f"[SAVED] {hist_path}")
    return hist_df


@torch.no_grad()
def evaluate_head_from_cached_features(
    model: VisiumHDLinearHeadModel,
    test_loader: DataLoader,
    test_ds: ViT_VisiumHD,
    args,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate the two-layer gene head on cached test features.
    Output files are kept compatible with the original script.
    """
    model.gene_head.to(device)
    model.gene_head.eval()

    num_test = len(test_ds)
    gene_names = test_ds.gene_set
    num_genes = len(gene_names)

    stats = init_gene_metric_stats(num_genes)
    cell_corr_all = []

    output_dir = args.output_dir
    ensure_dir(output_dir)

    no_scc_effective = bool(args.no_scc)

    need_pred_file = args.save_pred or (not no_scc_effective)
    pred_path = os.path.join(output_dir, "pred_test_8um.npy")
    tmp_pred_path = os.path.join(output_dir, "tmp_pred_for_scc.npy")
    active_pred_path = pred_path if args.save_pred else tmp_pred_path
    pred_memmap = None
    write_start = 0

    if need_pred_file:
        pred_memmap = np.lib.format.open_memmap(
            active_pred_path,
            mode="w+",
            dtype=np.float32,
            shape=(num_test, num_genes),
        )
        print(f"[INFO] Prediction memmap created: {active_pred_path}")

    pbar = tqdm(test_loader, desc="Evaluate cached test features")
    for features, exp in pbar:
        features = features.to(device, non_blocking=True).float()
        exp_np = exp.numpy().astype(np.float32)

        with autocast_context(args, device):
            pred = model.gene_head(features)

        pred_np = pred.detach().float().cpu().numpy().astype(np.float32)

        update_gene_metric_stats(stats, exp_np, pred_np)
        cell_corr_all.append(batch_cell_pcc(exp_np, pred_np))

        if pred_memmap is not None:
            bsz = pred_np.shape[0]
            pred_memmap[write_start:write_start + bsz] = pred_np
            write_start += bsz

    if pred_memmap is not None:
        pred_memmap.flush()
        del pred_memmap

    pcc, mse, mae = finalize_gene_metric_stats(stats)

    if no_scc_effective:
        scc = np.full_like(pcc, np.nan, dtype=np.float32)
    else:
        y_true_for_scc = test_ds.exp.numpy().astype(np.float32)
        y_pred_for_scc = np.load(active_pred_path, mmap_mode="r")
        scc = compute_spearman_per_gene(y_true_for_scc, y_pred_for_scc)

    cell_corr = np.concatenate(cell_corr_all, axis=0) if len(cell_corr_all) else np.array([], dtype=np.float32)
    valid_cell_corr = cell_corr[~np.isnan(cell_corr)]

    y_test = test_ds.exp.numpy().astype(np.float32)
    ind_heg50 = np.argsort(np.sum(y_test, axis=0))[-50:]
    ind_hvg50 = np.argsort(np.var(y_test, axis=0))[-50:]
    heg50_pcc = pcc[ind_heg50]
    hvg50_pcc = pcc[ind_hvg50]
    heg50_pcc = heg50_pcc[~np.isnan(heg50_pcc)]
    hvg50_pcc = hvg50_pcc[~np.isnan(hvg50_pcc)]

    per_gene = pd.DataFrame({
        "gene": gene_names,
        "PCC": pcc,
        "SCC": scc,
        "MSE": mse,
        "MAE": mae,
    })
    per_gene_path = os.path.join(output_dir, "per_gene_metrics_8um.csv")
    per_gene.to_csv(per_gene_path, index=False)

    summary = {
        "train_slide": args.train_slide,
        "test_slide": args.test_slide,
        "train_aug": False,
        "normal": bool(args.normal),
        "feature_cache": True,
        "feature_cache_dtype": str(getattr(args, "feature_cache_dtype", "float16")),
        "eval_name": "Final 8um test performance",
        "no_scc_effective": bool(no_scc_effective),
        "train_feature_path": args._resolved_train_feature_path,
        "test_feature_path": args._resolved_test_feature_path,
        "train_spots": int(args._train_spots),
        "test_spots": int(num_test),
        "num_genes": int(num_genes),
        "arch": args.arch,
        "feature_pool": "HFAM",
        "encoder_ckpt": args.encoder_ckpt,
        "head_structure": f"Linear({model.encoder_dim}, {model.head_hidden_dim}) -> Linear({model.head_hidden_dim}, {model.n_genes})",
        "head_batch_size": int(args.head_batch_size),
        "feature_batch_size": int(args.feature_batch_size),
        "max_epoch": int(args.max_epoch),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "mean_gene_PCC": float(np.nanmean(pcc)),
        "median_gene_PCC": float(np.nanmedian(pcc)),
        "max_gene_PCC": float(np.nanmax(pcc)),
        "median_HEG50_gene_PCC": float(np.nanmedian(heg50_pcc)) if len(heg50_pcc) else float("nan"),
        "median_HVG50_gene_PCC": float(np.nanmedian(hvg50_pcc)) if len(hvg50_pcc) else float("nan"),
        "mean_gene_SCC": float(np.nanmean(scc)),
        "median_gene_SCC": float(np.nanmedian(scc)),
        "mean_gene_MSE": float(np.nanmean(mse)),
        "median_gene_MSE": float(np.nanmedian(mse)),
        "mean_gene_MAE": float(np.nanmean(mae)),
        "median_gene_MAE": float(np.nanmedian(mae)),
        "median_cell_PCC": float(np.nanmedian(valid_cell_corr)) if len(valid_cell_corr) else float("nan"),
        "mean_cell_PCC": float(np.nanmean(valid_cell_corr)) if len(valid_cell_corr) else float("nan"),
    }

    summary_path = os.path.join(output_dir, "summary_metrics_8um.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    cell_path = os.path.join(output_dir, "cell_pcc_8um.npy")
    np.save(cell_path, cell_corr.astype(np.float32))

    if (not args.save_pred) and os.path.exists(tmp_pred_path):
        try:
            os.remove(tmp_pred_path)
            print(f"[INFO] Removed temporary prediction file: {tmp_pred_path}")
        except Exception as e:
            warnings.warn(f"Failed to remove temporary prediction file {tmp_pred_path}: {e}")

    print("\n" + "=" * 90)
    print("Final 8um test performance")
    print(f"Median cell PCC:      {summary['median_cell_PCC']:.6f}")
    print(f"Mean gene PCC:        {summary['mean_gene_PCC']:.6f}")
    print(f"Median gene PCC:      {summary['median_gene_PCC']:.6f}")
    print(f"Max gene PCC:         {summary['max_gene_PCC']:.6f}")
    print(f"Median HEG50 PCC:     {summary['median_HEG50_gene_PCC']:.6f}")
    print(f"Median HVG50 PCC:     {summary['median_HVG50_gene_PCC']:.6f}")
    print(f"Mean gene SCC:        {summary['mean_gene_SCC']:.6f}")
    print(f"Median gene SCC:      {summary['median_gene_SCC']:.6f}")
    print(f"Mean gene MSE:        {summary['mean_gene_MSE']:.6f}")
    print(f"Median gene MSE:      {summary['median_gene_MSE']:.6f}")
    print(f"Mean gene MAE:        {summary['mean_gene_MAE']:.6f}")
    print(f"Median gene MAE:      {summary['median_gene_MAE']:.6f}")
    print("=" * 90)
    print(f"[SAVED] {per_gene_path}")
    print(f"[SAVED] {summary_path}")
    print(f"[SAVED] {cell_path}")
    if args.save_pred:
        print(f"[SAVED] {pred_path}")

    return summary

# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train frozen iBOT-MambaVision + two-layer Linear head on VisiumHD 16um, "
            "test on 8um. This version first caches frozen encoder features and then "
            "trains only on feature vectors."
        )
    )

    # Data
    parser.add_argument("--config", type=str, default="configs/paths.yaml", help="YAML file containing dataset and checkpoint paths.")
    parser.add_argument(
        "--target_task",
        type=str,
        default="",
        choices=["hvg200", "marker10"],
        help="Prediction target preset. hvg200 and marker10 reproduce their respective reference settings.",
    )
    parser.add_argument("--base_dir", type=str, default="", help="Shared base dir. Used if train/test base dirs are not specified.")
    parser.add_argument("--train_base_dir", type=str, default="", help="Base dir containing the train expression CSV and subposition.parquet.")
    parser.add_argument("--test_base_dir", type=str, default="", help="Base dir containing the test expression CSV and subposition.parquet.")
    parser.add_argument("--train_slide", type=str, default="", help="Usually the 16um bin slide/folder name.")
    parser.add_argument("--test_slide", type=str, default="", help="Usually the 8um bin slide/folder name.")
    parser.add_argument("--train_he_tif", type=str, default="", help="H&E full-resolution tif for 16um train bins.")
    parser.add_argument("--test_he_tif", type=str, default="", help="H&E full-resolution tif for 8um test bins. If empty, use train_he_tif.")
    parser.add_argument("--hvg_file", type=str, default="", help=".npy file containing the ordered target gene names.")
    parser.add_argument("--expression_file", type=str, default="", help="Expression CSV filename inside each slide directory.")
    parser.add_argument(
        "--gene_mode",
        type=str,
        default="all",
        choices=["all", "first_k", "topk_plus_markers", "exclude_markers", "markers_only"],
        help=(
            "Target gene selection mode. all=original hvg_file; first_k=first --gene_num genes; "
            "topk_plus_markers=first --gene_num non-marker genes + markers; "
            "exclude_markers=all hvg_file genes except markers; markers_only=only markers."
        ),
    )
    parser.add_argument("--gene_num", type=int, default=0, help="K for first_k or topk_plus_markers. Example: 100 for HVG100+markers.")
    parser.add_argument("--marker_genes", type=str, default="", help="Comma-separated marker genes, e.g. ERBB2,MKI67. Optional.")
    parser.add_argument("--marker_gene_file", type=str, default="", help="Optional .npy/.txt/.csv/.tsv file containing marker gene names.")
    parser.add_argument(
        "--marker_from_hvg_tail",
        type=int,
        default=0,
        help="Use the last N genes from hvg_file as marker genes.",
    )
    parser.add_argument(
        "--expression_norm_scope",
        type=str,
        default="base",
        choices=["base", "target"],
        help=(
            "base: normalize expression over all genes in hvg_file, then subset target genes. "
            "target: normalize only over selected target genes. base is recommended for marker-only or HVG100+marker runs."
        ),
    )
    parser.add_argument("--crop_radius", type=int, default=112, help="Patch radius in pixels. r=112 gives 224x224 input.")
    parser.add_argument("--normal", type=str2bool, default=True, help="Whether to use ImageNet normalization after to_tensor.")
    parser.add_argument("--check_dataset", type=str2bool, default=True, help="Whether to run dataset sanity checks and print dataset statistics.")

    # Encoder and head
    parser.add_argument("--project_root", type=str, default="", help="Project root containing the `models` package. Optional if running from project root.")
    parser.add_argument("--arch", type=str, default="mamba_vision_base", choices=[
        "mamba_vision_tiny",
        "mamba_vision_t2",
        "mamba_vision_small",
        "mamba_vision_base",
        "mamba_vision_large",
        "mamba_vision_l2",
    ])
    parser.add_argument("--encoder_ckpt", type=str, default="", help="Path to the trained FM encoder checkpoint.")
    parser.add_argument("--strict_load", type=str2bool, default=True, help="Whether to strictly load the encoder checkpoint.")
    parser.add_argument("--head_hidden_dim", type=int, default=1024, help="Hidden dim of the two-layer Linear head.")
    parser.add_argument("--linear_head_ckpt", type=str, default="", help="Path to save/load the linear-head-only checkpoint. If empty, use output_dir/linear_head_ckpt/linear_head_last.pt")

    # Feature cache
    parser.add_argument("--feature_dir", type=str, default="", help="Directory for cached train/test feature .npy files. If empty, use output_dir/cached_features.")
    parser.add_argument("--train_feature_path", type=str, default="", help="Path to train cached features. If it exists, skip train encoding unless --force_extract_features True.")
    parser.add_argument("--test_feature_path", type=str, default="", help="Path to test cached features. If it exists, skip test encoding unless --force_extract_features True.")
    parser.add_argument("--force_extract_features", type=str2bool, default=False, help="True: re-extract features even if cached .npy files already exist.")
    parser.add_argument("--feature_cache_dtype", type=str, default="float16", choices=["float16", "float32"], help="Dtype used to save cached frozen-encoder features; float16 uses roughly half the disk space.")
    parser.add_argument("--feature_batch_size", type=int, default=64, help="Image batch size used only for frozen encoder feature extraction.")
    parser.add_argument("--head_batch_size", type=int, default=1024, help="Feature-vector batch size used for training/evaluating the gene head.")
    parser.add_argument("--feature_num_workers", type=int, default=0, help="DataLoader workers for cached-feature head training.")
    parser.add_argument("--load_train_features_to_ram", type=str2bool, default=False, help="Load train features fully into RAM instead of mmap.")
    parser.add_argument("--load_test_features_to_ram", type=str2bool, default=False, help="Load test features fully into RAM instead of mmap.")

    # Training
    parser.add_argument("--train_test", type=str2bool, default=True, help="True: train head and test. False: only test existing linear head.")
    parser.add_argument("--max_epoch", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr_step_size", type=int, default=50, help="StepLR step size. <=0 disables scheduler.")
    parser.add_argument("--lr_gamma", type=float, default=0.9)
    parser.add_argument("--grad_clip_norm", type=float, default=0.0)

    # Evaluation
    parser.add_argument("--no_scc", action="store_true", help="Skip Spearman correlation because it can be slow for large 8um data.")
    parser.add_argument("--save_pred", action="store_true", help="Save 8um predictions as pred_test_8um.npy.")

    # Runtime
    parser.add_argument("--tag", type=str, default="", help="Experiment name. If empty, generate automatically.")
    parser.add_argument("--output_dir", type=str, default="", help="Output folder. If empty, use results/<tag>.")
    parser.add_argument("--num_workers", type=int, default=8, help="Workers for image patch loading during feature extraction.")
    parser.add_argument("--device_num", type=int, default=0)
    parser.add_argument("--device", type=str, default="", help="Example: cuda:0. If empty, use cuda:<device_num> when available.")
    parser.add_argument("--amp", type=str, default="fp16", choices=["none", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=2024)

    args = parser.parse_args()

    config = load_path_config(args.config)
    task_config = config.get("visium_hd", {}) or {}
    args.target_task = args.target_task or task_config.get("target_task", "hvg200")
    args.base_dir = args.base_dir or task_config.get("base_dir", "")
    args.train_slide = args.train_slide or task_config.get("train_slide", "")
    args.test_slide = args.test_slide or task_config.get("test_slide", "")
    args.train_he_tif = args.train_he_tif or task_config.get("train_he_tif", task_config.get("he_tif", ""))
    args.test_he_tif = args.test_he_tif or task_config.get("test_he_tif", "")
    args.hvg_file = args.hvg_file or task_config.get("hvg_file", "")
    args.expression_file = args.expression_file or task_config.get("expression_file", "exp_HVG200_MARKER10.csv")
    args.feature_dir = args.feature_dir or task_config.get("feature_dir", "")
    args.train_feature_path = args.train_feature_path or task_config.get("train_feature_path", "")
    args.test_feature_path = args.test_feature_path or task_config.get("test_feature_path", "")
    args.encoder_ckpt = args.encoder_ckpt or config.get("checkpoint", "")
    apply_target_task_preset(args)

    required = {
        "train_slide": args.train_slide,
        "test_slide": args.test_slide,
        "train_he_tif": args.train_he_tif,
        "hvg_file": args.hvg_file,
        "encoder_ckpt": args.encoder_ckpt,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        parser.error(
            "Missing required paths: " + ", ".join(missing)
            + ". Set them in configs/paths.yaml or pass the corresponding command-line options."
        )

    seed_torch(args.seed)
    add_project_root(args.project_root)

    if args.train_base_dir == "":
        args.train_base_dir = args.base_dir
    if args.test_base_dir == "":
        args.test_base_dir = args.base_dir
    if args.train_base_dir == "" or args.test_base_dir == "":
        raise ValueError("Please provide --base_dir or both --train_base_dir and --test_base_dir.")
    if args.test_he_tif == "":
        args.test_he_tif = args.train_he_tif

    if args.tag is None or str(args.tag).strip() == "":
        args.tag = make_auto_tag(args)
        print(f"[INFO] Auto-generated tag: {args.tag}")
    else:
        print(f"[INFO] Using user-provided tag: {args.tag}")

    if args.output_dir is None or str(args.output_dir).strip() == "":
        args.output_dir = os.path.join("results", args.tag)
    ensure_dir(args.output_dir)
    save_run_config(args, args.output_dir)

    device = get_device(args)

    print("=" * 90)
    print("VisiumHD 16um -> 8um cross-resolution prediction")
    print("Setting: frozen FM encoder -> cached features -> two-layer Linear head")
    print("No PCA / No Ridge / No random augmentation for cached features")
    print(f"Train slide/bin:       {args.train_slide}")
    print(f"Test  slide/bin:       {args.test_slide}")
    print(f"target_task:           {args.target_task}")
    print(f"arch:                  {args.arch}")
    print("feature_pool:          HFAM")
    print(f"normal:                {args.normal}")
    print(f"feature_batch_size:    {args.feature_batch_size}")
    print(f"head_batch_size:       {args.head_batch_size}")
    print(f"feature_cache_dtype:   {args.feature_cache_dtype}")
    print(f"gene_mode:             {args.gene_mode}")
    print(f"gene_num:              {args.gene_num}")
    print(f"marker_genes:          {args.marker_genes}")
    print(f"marker_gene_file:      {args.marker_gene_file}")
    print(f"marker_from_hvg_tail:  {args.marker_from_hvg_tail}")
    print(f"expression_norm_scope: {args.expression_norm_scope}")
    print(f"device:                {device}")
    print("=" * 90)

    train_ds = ViT_VisiumHD(
        base_dir=args.train_base_dir,
        he_tif_path=args.train_he_tif,
        hvg_file=args.hvg_file,
        slide_name=args.train_slide,
        expression_file=args.expression_file,
        r=args.crop_radius,
        train=True,
        train_aug=False,
        normal=args.normal,
        check_dataset=args.check_dataset,
        gene_mode=args.gene_mode,
        gene_num=args.gene_num,
        marker_genes=args.marker_genes,
        marker_gene_file=args.marker_gene_file,
        marker_from_hvg_tail=args.marker_from_hvg_tail,
        expression_norm_scope=args.expression_norm_scope,
    )
    test_ds = ViT_VisiumHD(
        base_dir=args.test_base_dir,
        he_tif_path=args.test_he_tif,
        hvg_file=args.hvg_file,
        slide_name=args.test_slide,
        expression_file=args.expression_file,
        r=args.crop_radius,
        train=False,
        train_aug=False,
        normal=args.normal,
        check_dataset=args.check_dataset,
        gene_mode=args.gene_mode,
        gene_num=args.gene_num,
        marker_genes=args.marker_genes,
        marker_gene_file=args.marker_gene_file,
        marker_from_hvg_tail=args.marker_from_hvg_tail,
        expression_norm_scope=args.expression_norm_scope,
    )

    if train_ds.gene_set != test_ds.gene_set:
        raise ValueError("Train/test gene order is not identical. Please use the same hvg_file and gene order.")

    args._train_spots = int(len(train_ds))
    print(f"[INFO] Train spots: {len(train_ds):,}, Test spots: {len(test_ds):,}, Genes: {len(train_ds.gene_set):,}")

    model = build_model(args, n_genes=len(train_ds.gene_set))
    print("=" * 90)
    print(f"[MODEL] encoder_dim: {model.encoder_dim}")
    print(f"[MODEL] gene_head: Linear({model.encoder_dim}, {model.head_hidden_dim}) -> Linear({model.head_hidden_dim}, {model.n_genes})")
    print("=" * 90)

    train_feature_path, test_feature_path = load_or_extract_cached_features(
        model=model,
        train_ds=train_ds,
        test_ds=test_ds,
        args=args,
        device=device,
    )
    args._resolved_train_feature_path = train_feature_path
    args._resolved_test_feature_path = test_feature_path

    train_feature_ds = CachedFeatureDataset(
        feature_path=train_feature_path,
        exp_tensor=train_ds.exp,
        load_to_ram=args.load_train_features_to_ram,
    )
    test_feature_ds = CachedFeatureDataset(
        feature_path=test_feature_path,
        exp_tensor=test_ds.exp,
        load_to_ram=args.load_test_features_to_ram,
    )

    train_feature_loader = make_feature_training_loader(train_feature_ds, args, train=True)
    test_feature_loader = make_feature_training_loader(test_feature_ds, args, train=False)

    # The encoder is no longer needed after feature caching. Keep only the head on GPU.
    model.ibot_mamba.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.train_test:
        print("[INFO] Start training two-layer Linear head from cached feature vectors.")
        train_head_from_cached_features(
            model=model,
            train_loader=train_feature_loader,
            args=args,
            device=device,
        )
        save_linear_head_checkpoint(model, args, gene_names=train_ds.gene_set)
    else:
        print("[INFO] train_test=False. Skip training and load existing linear head checkpoint.")
        model = load_linear_head_checkpoint(model, args)

    print("[INFO] Start testing on cached high-resolution / 8um features.")
    evaluate_head_from_cached_features(model, test_feature_loader, test_ds, args, device)

    print("All finished.")


if __name__ == "__main__":
    main()
