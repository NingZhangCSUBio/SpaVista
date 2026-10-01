#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tumor detection with a frozen SpaVista encoder and classification head.

Model: SpaVista with HFAM.

Dataset labels:
  tumor_from_A == 'non'   -> 0
  tumor_from_A == 'tumor' -> 1

Metric rules:
  - Training: CrossEntropyLoss(logits, labels)
  - ROC / AUROC / AUPRC: use softmax(logits)[:, 1], i.e. tumor probability
  - Accuracy / Balanced Accuracy / Precision / Sensitivity / Specificity / F1:
      default uses tumor_probability >= 0.5
      set --use_threshold_for_metrics False to use argmax(softmax) instead

Outputs:
  results/<tag>/
    all_model_metrics.csv
    roc_curves.png
    roc_curves.pdf
    <MODEL>/
      linear_probe_last.pt
      train_log.csv
      test_predictions.csv
      metrics.csv
      confusion_matrix.csv
      confusion_matrix.png
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from tqdm import tqdm

try:
    import timm
except Exception as e:
    raise ImportError("This script needs timm. Install it with: pip install timm") from e

try:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        average_precision_score,
        roc_curve,
        confusion_matrix,
    )
except Exception as e:
    raise ImportError("This script needs scikit-learn. Install it with: pip install scikit-learn") from e

import matplotlib.pyplot as plt

from spavista.config import load_path_config

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
# =============================================================================
# Basic utils
# =============================================================================
def str2bool(v):
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


def capture_rng_state() -> Dict:
    """Capture RNG state so feature extraction cannot affect head initialization."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Dict):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def check_file(path: str, name: str):
    if path is None or str(path).strip() == "":
        raise ValueError(f"{name} is empty.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find {name}: {path}")


def add_project_root(project_root: str):
    if project_root and str(project_root).strip() != "":
        project_root = os.path.abspath(project_root)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        print(f"[INFO] Added project_root to sys.path: {project_root}")


def save_run_config(args, save_path: str):
    ensure_dir(save_path)
    run_info = {
        "saved_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version,
        "torch": torch.__version__,
        "timm": getattr(timm, "__version__", "unknown"),
        "command": " ".join(sys.argv),
        "args": vars(args),
    }
    with open(os.path.join(save_path, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=4, ensure_ascii=False)
    with open(os.path.join(save_path, "run_command.txt"), "w", encoding="utf-8") as f:
        f.write(" ".join(sys.argv) + "\n")


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


def parse_comma_list(x: str) -> List[str]:
    if x is None or str(x).strip() == "":
        return []
    return [v.strip() for v in str(x).split(",") if v.strip() != ""]


def sanitize_name(x: str) -> str:
    x = str(x)
    for ch in ["/", "\\", ":", " ", "\t", "\n", ",", ";"]:
        x = x.replace(ch, "_")
    return x


# =============================================================================
# Dataset
# =============================================================================
class STNetDataClassifyLazy(Dataset):
    """
    Lazy version of the STNet tumor-detection dataset.

    Init records samples as (tif_path, x, y, label, slide_id). Patches are cropped in
    __getitem__. This avoids appending all patches into RAM during construction.
    """

    def __init__(
        self,
        root_dir: str,
        test_folders: List[str],
        train: bool = True,
        patch_size: int = 224,
        train_aug: bool = True,
        normal: bool = True,
        check_patch: bool = True,
        cache_images: bool = True,
        verbose: bool = True,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.test_folders = set([str(x) for x in test_folders])
        self.train = bool(train)
        self.patch_size = int(patch_size)
        self.r = self.patch_size // 2
        self.train_aug = bool(train_aug)
        self.normal = bool(normal)
        self.check_patch = bool(check_patch)
        self.cache_images = bool(cache_images)
        self.verbose = bool(verbose)

        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"Cannot find root_dir: {self.root_dir}")

        all_folders = sorted([
            f for f in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, f))
        ])
        self.selected_folders = (
            [f for f in all_folders if f not in self.test_folders]
            if self.train else
            [f for f in all_folders if f in self.test_folders]
        )

        if self.verbose:
            print("Training slides:" if self.train else "Testing slides:", self.selected_folders)

        self.samples: List[Dict] = []
        self.slide_counts: Dict[str, int] = {}
        self.label_counts = {0: 0, 1: 0}
        self._image_size_cache: Dict[str, Tuple[int, int]] = {}
        self._image_cache: Dict[str, np.ndarray] = {}
        self._build_index()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid samples found. train={self.train}, root_dir={self.root_dir}, "
                f"selected_folders={self.selected_folders}"
            )

        if self.verbose:
            print("-" * 80)
            print(f"[DATASET] train={self.train}")
            print(f"[DATASET] samples={len(self.samples):,}")
            print(f"[DATASET] label counts: non=0 -> {self.label_counts[0]:,}, tumor=1 -> {self.label_counts[1]:,}")
            print(f"[DATASET] train_aug={self.train_aug}, normal={self.normal}, patch_size={self.patch_size}")
            print("-" * 80)

    def _image_size(self, tif_path: str) -> Tuple[int, int]:
        if tif_path not in self._image_size_cache:
            with Image.open(tif_path) as img:
                self._image_size_cache[tif_path] = img.size  # (W, H)
        return self._image_size_cache[tif_path]

    def _build_index(self):
        print("Loading classification sample index...")
        for folder in self.selected_folders:
            folder_path = os.path.join(self.root_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            files = os.listdir(folder_path)
            tsv_files = sorted([f for f in files if f.lower().endswith(".tsv")])
            tif_files = sorted([f for f in files if f.lower().endswith((".tif", ".tiff"))])

            for tsv_file in tsv_files:
                tsv_identifier = tsv_file[2:10]
                matched_tif = next((f for f in tif_files if f[5:13] == tsv_identifier), None)
                if matched_tif is None:
                    print(f"[WARN] No matched tif for {os.path.join(folder_path, tsv_file)}")
                    continue

                tsv_path = os.path.join(folder_path, tsv_file)
                tif_path = os.path.join(folder_path, matched_tif)

                try:
                    tsv_data = pd.read_csv(tsv_path, sep="\t")
                except Exception as e:
                    print(f"[WARN] Failed to read {tsv_path}: {repr(e)}")
                    continue

                required_cols = ["pixel_x", "pixel_y", "tumor_from_A"]
                missing = [c for c in required_cols if c not in tsv_data.columns]
                if missing:
                    print(f"[WARN] {tsv_path} missing columns {missing}. Skip.")
                    continue

                try:
                    W, H = self._image_size(tif_path)
                except Exception as e:
                    print(f"[WARN] Failed to read image size {tif_path}: {repr(e)}")
                    continue

                added = 0
                for _, row in tsv_data.iterrows():
                    tumor = str(row["tumor_from_A"])
                    if tumor == "non":
                        label = 0
                    elif tumor == "tumor":
                        label = 1
                    else:
                        print(f"[WARN] Unknown class {tumor}, file {tsv_path}, skip.")
                        continue

                    x = int(row["pixel_x"])
                    y = int(row["pixel_y"])

                    if self.check_patch:
                        if (x - self.r < 0) or (x + self.r > W) or (y - self.r < 0) or (y + self.r > H):
                            continue

                    self.samples.append({
                        "tif_path": tif_path,
                        "x": x,
                        "y": y,
                        "label": int(label),
                        "slide": str(folder),
                        "tsv_path": tsv_path,
                    })
                    self.label_counts[int(label)] += 1
                    added += 1

                self.slide_counts[folder] = self.slide_counts.get(folder, 0) + added

        if self.verbose:
            print("[DATASET] slide sample counts:")
            for k, v in sorted(self.slide_counts.items()):
                print(f"  {k}: {v:,}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image_array(self, tif_path: str) -> np.ndarray:
        if self.cache_images and tif_path in self._image_cache:
            return self._image_cache[tif_path]
        print("loading-----", tif_path)
        img = Image.open(tif_path).convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)
        if self.cache_images:
            self._image_cache[tif_path] = arr
        return arr

    def _augment_pil(self, image: Image.Image) -> Image.Image:
        if random.random() > 0.5:
            image = TF.hflip(image)
        if random.random() > 0.5:
            image = TF.vflip(image)
        angle = random.choice([180, 90, 0, -90])
        image = TF.rotate(image, angle)
        return image

    def transform(self, patch: np.ndarray):
        image = Image.fromarray(patch)
        if self.train and self.train_aug:
            image = self._augment_pil(image)
        image = TF.to_tensor(image)
        if self.normal:
            image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return image

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        img = self._load_image_array(item["tif_path"])
        x, y = item["x"], item["y"]
        patch = img[(y - self.r):(y + self.r), (x - self.r):(x + self.r), :]
        if patch.shape[:2] != (self.patch_size, self.patch_size):
            patch = np.asarray(Image.fromarray(patch).resize((self.patch_size, self.patch_size)), dtype=np.uint8)
        patch = self.transform(patch)
        label = torch.tensor(item["label"], dtype=torch.long)
        return patch, label


# =============================================================================
# Checkpoint helpers
# =============================================================================
def _extract_state_dict(ckpt):
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint should be dict-like.")
    for key in ["state_dict", "model", "teacher", "student", "module", "backbone", "encoder"]:
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    return ckpt


def _strip_common_prefixes(state_dict):
    new_state_dict = {}
    prefixes = [
        "module.backbone.", "module.encoder.", "module.",
        "backbone.", "encoder.", "student.", "teacher.", "model.",
        "ibot_mamba.encoder.", "ibot_mamba.",
    ]
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


def load_checkpoint_to_model(model: nn.Module, ckpt_path: str, strict: bool = True):
    check_file(ckpt_path, "checkpoint")
    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = _extract_state_dict(ckpt)
    try:
        msg = model.load_state_dict(sd, strict=strict)
        print(f"[INFO] Loaded with original keys. strict={strict}")
        print(f"[INFO] missing keys: {getattr(msg, 'missing_keys', [])}")
        print(f"[INFO] unexpected keys: {getattr(msg, 'unexpected_keys', [])}")
        return
    except RuntimeError as e:
        print(f"[WARN] Direct load failed: {repr(e)}")
        print("[WARN] Trying stripped common prefixes...")
    sd = _strip_common_prefixes(sd)
    msg = model.load_state_dict(sd, strict=strict)
    print(f"[INFO] Loaded with stripped keys. strict={strict}")
    print(f"[INFO] missing keys: {getattr(msg, 'missing_keys', [])}")
    print(f"[INFO] unexpected keys: {getattr(msg, 'unexpected_keys', [])}")


# =============================================================================
# SpaVista / iBOT-MambaVision encoder
# =============================================================================
class SpaVistaEncoder(nn.Module):
    VALID_POOLS = ["HFAM"]

    def __init__(self, checkpoint_path: str, arch: str, feature_pool: str, strict: bool):
        super().__init__()
        try:
            from spavista import models
        except Exception as e:
            raise ImportError("Cannot import `models`. Run from project root or set --project_root.") from e
        if not hasattr(models, arch):
            raise ValueError(f"Unknown MambaVision arch: {arch}")
        if feature_pool not in self.VALID_POOLS:
            raise ValueError(f"Unknown feature_pool={feature_pool}")
        self.arch = arch
        self.feature_pool = feature_pool
        self.encoder = getattr(models, arch)(pretrained=False, return_all_tokens=False, masked_im_modeling=False)
        if hasattr(self.encoder, "embed_dim"):
            self.encoder_dim = int(self.encoder.embed_dim)
        elif hasattr(self.encoder, "num_features"):
            self.encoder_dim = int(self.encoder.num_features)
        else:
            raise AttributeError(f"Cannot infer output dim from SpaVista arch={arch}")
        self.output_dim = self.encoder_dim * 2
        load_checkpoint_to_model(self.encoder, checkpoint_path, strict=strict)

    @staticmethod
    def _to_bchw(x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise RuntimeError(f"Expected 4D feature map, got {tuple(x.shape)}")
        common_channels = {64,80,96,128,160,192,256,320,384,512,640,768,784,1024,1536,1568,2048}
        if x.shape[1] not in common_channels and x.shape[-1] in common_channels:
            x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def _get_mambavision_model(self):
        model = self.encoder.model if hasattr(self.encoder, "model") else self.encoder
        if not hasattr(model, "patch_embed") or not hasattr(model, "levels"):
            raise RuntimeError("Cannot find model.patch_embed or model.levels.")
        return model

    def _forward_all_levels(self, x):
        model = self._get_mambavision_model()
        x = model.patch_embed(x)
        outs = []
        for level in model.levels:
            x = level(x)
            if isinstance(x, (tuple, list)):
                x = x[0]
            outs.append(self._to_bchw(x))
        return outs

    def _forward_late_stage_maps(self, x):
        model = self._get_mambavision_model()
        outs = self._forward_all_levels(x)
        if len(outs) < 4:
            raise RuntimeError(f"Expected 4 levels, got {len(outs)}")
        feat_a, feat_b = outs[2], outs[3]
        if hasattr(model, "norm") and model.norm is not None:
            try:
                feat_b = model.norm(feat_b)
            except Exception:
                pass
        return self._to_bchw(feat_a), self._to_bchw(feat_b)

    @staticmethod
    def _gap(feat):
        return F.adaptive_avg_pool2d(feat, output_size=1).flatten(1)

    @staticmethod
    def _branch_l2(feat, eps=1e-12):
        return F.normalize(feat, p=2, dim=1, eps=eps)

    @staticmethod
    def _signed_sqrt(feat, eps=1e-12):
        return torch.sign(feat) * torch.sqrt(torch.clamp(torch.abs(feat), min=eps))

    @staticmethod
    def _crow_pool(feat, eps=1e-6):
        if feat.dim() != 4:
            raise RuntimeError(f"Expected [B,C,H,W], got {tuple(feat.shape)}")
        _, _, H, W = feat.shape
        feat_abs = torch.abs(feat)
        spatial = feat_abs.sum(dim=1, keepdim=True)
        spatial = spatial / (spatial.mean(dim=(2,3), keepdim=True) + eps)
        ch_mean = feat_abs.mean(dim=(2,3), keepdim=True)
        active = (feat_abs > ch_mean).float().sum(dim=(2,3), keepdim=True)
        channel = torch.log((float(H * W) + eps) / (active + eps))
        channel = channel / (channel.mean(dim=1, keepdim=True) + eps)
        return (feat * spatial * channel).mean(dim=(2,3))

    def _power_l2_readout(self, feat_a, feat_b):
        fa = self._branch_l2(self._signed_sqrt(self._gap(feat_a)))
        fb = self._branch_l2(self._signed_sqrt(self._gap(feat_b)))
        return torch.cat([fa, fb], dim=1)

    def _crow_readout(self, feat_a, feat_b):
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


def build_encoder(args, model_key="spavista"):
    """Build the frozen SpaVista encoder used for tumor detection."""
    if str(model_key).lower() != "spavista":
        raise ValueError("Only the SpaVista encoder is supported in this repository.")
    check_file(args.spavista_ckpt, "SpaVista checkpoint")
    encoder = SpaVistaEncoder(
        checkpoint_path=args.spavista_ckpt,
        arch=args.spavista_arch,
        feature_pool="HFAM",
        strict=args.strict_load,
    )
    return encoder, encoder.output_dim, "SpaVista"


class ClassificationHead(nn.Module):
    """Linear or two-layer classifier trained on frozen SpaVista features."""

    def __init__(self, feature_dim: int, num_classes: int = 2, head_type: str = "linear", hidden_dim: int = 1024):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.head_type = canonical_head_type(head_type)
        if self.head_type == "linear":
            self.layers = nn.Linear(self.feature_dim, self.num_classes)
        else:
            self.layers = nn.Sequential(
                nn.Linear(self.feature_dim, self.hidden_dim),
                nn.Linear(self.hidden_dim, self.num_classes),
            )

    def forward(self, features):
        return self.layers(features)

    def structure_string(self):
        if self.head_type == "linear":
            return f"Linear({self.feature_dim}, {self.num_classes})"
        return (
            f"Linear({self.feature_dim}, {self.hidden_dim}) -> "
            f"Linear({self.hidden_dim}, {self.num_classes})"
        )


def canonical_head_type(head_type: str) -> str:
    h = str(head_type).lower().replace("-", "_")
    if h in ["linear", "single", "single_linear", "one_layer", "1layer", "1_layer"]:
        return "linear"
    if h in ["mlp1024", "two_layer", "two_layer_linear", "2layer", "2_layer", "double", "two"]:
        return "mlp1024"
    raise ValueError("--head_type must be one of: linear, mlp1024")


def make_image_loader(dataset: Dataset, args):
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    return DataLoader(
        dataset,
        batch_size=int(args.feature_batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=args.num_workers > 0,
    )


def make_feature_loader(dataset: Dataset, args, train: bool):
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    return DataLoader(
        dataset,
        batch_size=int(args.head_batch_size),
        shuffle=bool(train),
        num_workers=int(args.feature_num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=int(args.feature_num_workers) > 0,
    )


def get_numpy_dtype(dtype_name: str):
    dtype_name = str(dtype_name).lower()
    if dtype_name in ["float16", "fp16", "half"]:
        return np.float16
    if dtype_name in ["float32", "fp32", "single"]:
        return np.float32
    raise ValueError("--feature_cache_dtype must be float16 or float32")


def _path_meta(path: str) -> Dict:
    path = os.path.abspath(os.path.expanduser(str(path))) if path else ""
    if path and os.path.exists(path):
        st = os.stat(path)
        return {"path": path, "size": int(st.st_size), "mtime": int(st.st_mtime)}
    return {"path": path, "size": None, "mtime": None}


def _model_cache_signature(args, canonical_name: str, feature_dim: int):
    return {
        "model": "SpaVista",
        "feature_dim": int(feature_dim),
        "arch": args.spavista_arch,
        "readout": "HFAM",
        "checkpoint": _path_meta(args.spavista_ckpt),
    }


def feature_cache_key(args, canonical_name: str, feature_dim: int, split_name: str) -> str:
    import hashlib
    payload = {
        "split_name": split_name,
        "root_dir": os.path.abspath(os.path.expanduser(str(args.root_dir))),
        "test_folders": parse_comma_list(args.test_folders),
        "patch_size": int(args.patch_size),
        "normal": bool(args.normal),
        "check_patch": bool(args.check_patch),
        "feature_cache_dtype": str(args.feature_cache_dtype),
        "model_signature": _model_cache_signature(args, canonical_name, feature_dim),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.md5(raw).hexdigest()[:12]


def default_feature_paths(args, canonical_name: str, feature_dim: int, split_name: str) -> Tuple[str, str, str]:
    if args.feature_cache_dir is None or str(args.feature_cache_dir).strip() == "":
        feature_dir = os.path.join("data", "tumor_detection", "TRISTdata", "feature_cache")
    else:
        feature_dir = args.feature_cache_dir
    ensure_dir(feature_dir)

    key = feature_cache_key(args, canonical_name, feature_dim, split_name)
    model_tag = sanitize_name(canonical_name)
    suffix = (
        f"{model_tag}_{split_name}_"
        f"ps{int(args.patch_size)}_"
        f"normal{int(bool(args.normal))}_"
        f"dtype-{sanitize_name(args.feature_cache_dtype)}_"
        f"{key}"
    )
    feature_path = os.path.join(feature_dir, f"features_{suffix}.npy")
    label_path = os.path.join(feature_dir, f"labels_{suffix}.npy")
    meta_path = os.path.join(feature_dir, f"meta_{suffix}.json")
    return feature_path, label_path, meta_path


def resolve_feature_paths(args, canonical_name: str, feature_dim: int, split_name: str) -> Tuple[str, str, str]:
    default_feature_path, default_label_path, default_meta_path = default_feature_paths(args, canonical_name, feature_dim, split_name)
    if split_name == "train" and args.train_feature_path:
        feature_path = args.train_feature_path
        label_path = args.train_label_path if args.train_label_path else feature_path + ".labels.npy"
        meta_path = feature_path + ".meta.json"
    elif split_name == "test" and args.test_feature_path:
        feature_path = args.test_feature_path
        label_path = args.test_label_path if args.test_label_path else feature_path + ".labels.npy"
        meta_path = feature_path + ".meta.json"
    elif os.path.exists(default_feature_path) and os.path.exists(default_label_path):
        feature_path, label_path, meta_path = default_feature_path, default_label_path, default_meta_path
    else:
        feature_path, label_path, meta_path = default_feature_path, default_label_path, default_meta_path
        cache_dir = os.path.dirname(default_feature_path)
        prefix = f"features_{sanitize_name(canonical_name)}_{split_name}_"
        compatible = []
        for name in os.listdir(cache_dir):
            if not (name.startswith(prefix) and name.endswith(".npy")):
                continue
            suffix = name[len("features_"):-4]
            candidate_feature = os.path.join(cache_dir, name)
            candidate_label = os.path.join(cache_dir, f"labels_{suffix}.npy")
            candidate_meta = os.path.join(cache_dir, f"meta_{suffix}.json")
            if not (os.path.isfile(candidate_label) and os.path.isfile(candidate_meta)):
                continue
            try:
                with open(candidate_meta, "r", encoding="utf-8") as handle:
                    meta = json.load(handle)
                model_meta = meta.get("model_signature", {})
                valid = (
                    meta.get("model") == canonical_name
                    and meta.get("split_name") == split_name
                    and int(meta.get("feature_dim", -1)) == int(feature_dim)
                    and int(meta.get("patch_size", -1)) == int(args.patch_size)
                    and bool(meta.get("normal")) == bool(args.normal)
                    and [str(value) for value in meta.get("test_folders", [])] == parse_comma_list(args.test_folders)
                    and model_meta.get("arch") == args.spavista_arch
                    and model_meta.get("readout") == "HFAM"
                    and model_meta.get("checkpoint", {}).get("size") == os.stat(args.spavista_ckpt).st_size
                )
                if valid:
                    compatible.append((candidate_feature, candidate_label, candidate_meta))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        if len(compatible) > 1:
            raise ValueError(f"Multiple compatible {split_name} caches found; set an explicit feature path: {compatible}")
        if compatible:
            feature_path, label_path, meta_path = compatible[0]
    ensure_dir(os.path.dirname(feature_path))
    ensure_dir(os.path.dirname(label_path))
    ensure_dir(os.path.dirname(meta_path))
    return feature_path, label_path, meta_path


def validate_feature_cache(feature_path: str, label_path: str, expected_n: int, expected_dim: int, split_name: str, expected_labels=None):
    check_file(feature_path, f"{split_name} cached features")
    check_file(label_path, f"{split_name} cached labels")
    features = np.load(feature_path, mmap_mode="r")
    labels = np.load(label_path, mmap_mode="r")
    if features.ndim != 2:
        raise ValueError(f"{split_name} features should be [N,D], got {features.shape}")
    if int(features.shape[0]) != int(expected_n):
        raise ValueError(f"{split_name} features N={features.shape[0]}, expected {expected_n}. Path: {feature_path}")
    if int(features.shape[1]) != int(expected_dim):
        raise ValueError(f"{split_name} features D={features.shape[1]}, expected {expected_dim}. Path: {feature_path}")
    if labels.ndim != 1:
        raise ValueError(f"{split_name} labels should be [N], got {labels.shape}")
    if int(labels.shape[0]) != int(expected_n):
        raise ValueError(f"{split_name} labels N={labels.shape[0]}, expected {expected_n}. Path: {label_path}")
    if expected_labels is not None and not np.array_equal(labels, np.asarray(expected_labels)):
        raise ValueError(f"{split_name} cached labels do not match the current dataset sample order: {label_path}")
    print(f"[CACHE] Use existing {split_name} features: {feature_path}, shape={features.shape}, dtype={features.dtype}")
    print(f"[CACHE] Use existing {split_name} labels:   {label_path}, shape={labels.shape}, dtype={labels.dtype}")
    return feature_path, label_path


@torch.no_grad()
def extract_features_to_cache(encoder: nn.Module, dataset: Dataset, args, device: torch.device, canonical_name: str, feature_dim: int, split_name: str, feature_path: str, label_path: str, meta_path: str):
    ensure_dir(os.path.dirname(feature_path))
    ensure_dir(os.path.dirname(label_path))
    ensure_dir(os.path.dirname(meta_path))

    encoder.to(device)
    encoder.eval()
    dtype = get_numpy_dtype(args.feature_cache_dtype)
    num_samples = len(dataset)
    features_mm = np.lib.format.open_memmap(
        feature_path,
        mode="w+",
        dtype=dtype,
        shape=(int(num_samples), int(feature_dim)),
    )
    labels_np = np.zeros((int(num_samples),), dtype=np.int64)

    loader = make_image_loader(dataset, args)
    write_start = 0
    pbar = tqdm(loader, desc=f"[{canonical_name}] Extract {split_name} features")
    for patches, labels in pbar:
        patches = patches.to(device, non_blocking=True)
        with autocast_context(args, device):
            h = encoder(patches)
        h_np = h.detach().float().cpu().numpy().astype(dtype, copy=False)
        bsz = int(h_np.shape[0])
        features_mm[write_start:write_start + bsz] = h_np
        labels_np[write_start:write_start + bsz] = labels.detach().cpu().numpy().astype(np.int64)
        write_start += bsz

    if write_start != num_samples:
        raise RuntimeError(f"Feature extraction wrote {write_start} samples, expected {num_samples}.")

    features_mm.flush()
    del features_mm
    np.save(label_path, labels_np)

    meta = {
        "saved_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": canonical_name,
        "split_name": split_name,
        "feature_path": feature_path,
        "label_path": label_path,
        "num_samples": int(num_samples),
        "feature_dim": int(feature_dim),
        "feature_dtype": str(args.feature_cache_dtype),
        "root_dir": os.path.abspath(os.path.expanduser(str(args.root_dir))),
        "test_folders": parse_comma_list(args.test_folders),
        "patch_size": int(args.patch_size),
        "normal": bool(args.normal),
        "train_aug": False,
        "model_signature": _model_cache_signature(args, canonical_name, feature_dim),
        "note": "Frozen encoder features are extracted deterministically with train_aug=False. During cached-feature training, image random augmentation is not applied.",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)

    print(f"[SAVED] {feature_path}")
    print(f"[SAVED] {label_path}")
    print(f"[SAVED] {meta_path}")
    return feature_path, label_path


def load_or_extract_feature_cache(encoder: nn.Module, train_ds: Dataset, test_ds: Dataset, args, device: torch.device, canonical_name: str, feature_dim: int):
    paths = {}
    for split_name, dataset in [("train", train_ds), ("test", test_ds)]:
        feature_path, label_path, meta_path = resolve_feature_paths(args, canonical_name, feature_dim, split_name)
        need_extract = bool(args.force_extract_features) or (not os.path.exists(feature_path)) or (not os.path.exists(label_path))
        if need_extract:
            print(f"[CACHE] Extracting {split_name} features because cache is missing or --force_extract_features True.")
            extract_features_to_cache(
                encoder=encoder,
                dataset=dataset,
                args=args,
                device=device,
                canonical_name=canonical_name,
                feature_dim=feature_dim,
                split_name=split_name,
                feature_path=feature_path,
                label_path=label_path,
                meta_path=meta_path,
            )
        else:
            validate_feature_cache(
                feature_path, label_path, len(dataset), feature_dim, split_name,
                expected_labels=[sample["label"] for sample in dataset.samples],
            )
        paths[split_name] = (feature_path, label_path)
    return paths["train"], paths["test"]


class CachedFeatureDataset(Dataset):
    def __init__(self, feature_path: str, label_path: str, load_to_ram: bool = True):
        self.feature_path = feature_path
        self.label_path = label_path
        mmap_mode = None if load_to_ram else "r"
        self.features = np.load(feature_path, mmap_mode=mmap_mode)
        self.labels = np.load(label_path, mmap_mode=mmap_mode)
        if self.features.ndim != 2:
            raise ValueError(f"features should be [N,D], got {self.features.shape}")
        if self.labels.ndim != 1 or self.labels.shape[0] != self.features.shape[0]:
            raise ValueError(f"labels should be [N] and match features, got features={self.features.shape}, labels={self.labels.shape}")

    def __len__(self):
        return int(self.features.shape[0])

    def __getitem__(self, idx: int):
        feat = torch.from_numpy(np.asarray(self.features[idx]))
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return feat, label


def get_class_weights_from_labels(labels, device: torch.device):
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.array([(labels == 0).sum(), (labels == 1).sum()], dtype=np.float32)
    if np.any(counts <= 0):
        return None
    weights = counts.sum() / (2.0 * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def get_class_weights(train_ds: STNetDataClassifyLazy, device: torch.device):
    counts = np.array([train_ds.label_counts[0], train_ds.label_counts[1]], dtype=np.float32)
    if np.any(counts <= 0):
        return None
    weights = counts.sum() / (2.0 * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def compute_binary_metrics(y_true: np.ndarray, y_score: np.ndarray, y_pred_argmax: np.ndarray, args, model_name: str) -> Dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float32)
    y_pred_argmax = np.asarray(y_pred_argmax, dtype=np.int64)
    y_pred_threshold = (y_score >= float(args.threshold)).astype(np.int64)
    y_pred_metric = y_pred_threshold if args.use_threshold_for_metrics else y_pred_argmax
    cm = confusion_matrix(y_true, y_pred_metric, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "model": model_name,
        "head_type": canonical_head_type(args.head_type),
        "head_hidden_dim": int(args.head_hidden_dim),
        "n_test": int(len(y_true)),
        "n_non": int((y_true == 0).sum()),
        "n_tumor": int((y_true == 1).sum()),
        "threshold": float(args.threshold),
        "metric_prediction_rule": "threshold_tumor_prob" if args.use_threshold_for_metrics else "argmax_softmax",
        "accuracy": float(accuracy_score(y_true, y_pred_metric)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred_metric)),
        "precision": float(precision_score(y_true, y_pred_metric, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred_metric, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan"),
        "f1": float(f1_score(y_true, y_pred_metric, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else float("nan"),
        "auprc": float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    return metrics


def save_eval_outputs(y_true: np.ndarray, y_score: np.ndarray, y_pred_argmax: np.ndarray, metrics: Dict, model_name: str, args, save_dir: str):
    ensure_dir(save_dir)
    y_pred_threshold = (y_score >= float(args.threshold)).astype(np.int64)
    y_pred_metric = y_pred_threshold if args.use_threshold_for_metrics else y_pred_argmax
    cm = confusion_matrix(y_true, y_pred_metric, labels=[0, 1])
    fpr, tpr, _ = roc_curve(y_true, y_score) if len(np.unique(y_true)) == 2 else (np.array([np.nan]), np.array([np.nan]), None)
    pd.DataFrame({
        "y_true": y_true,
        "tumor_probability": y_score,
        "y_pred_argmax": y_pred_argmax,
        f"y_pred_threshold_{float(args.threshold):.3f}": y_pred_threshold,
        "y_pred_for_metrics": y_pred_metric,
    }).to_csv(os.path.join(save_dir, "test_predictions.csv"), index=False)
    pd.DataFrame([metrics]).to_csv(os.path.join(save_dir, "metrics.csv"), index=False)
    pd.DataFrame(cm, index=["true_non", "true_tumor"], columns=["pred_non", "pred_tumor"]).to_csv(os.path.join(save_dir, "confusion_matrix.csv"))
    plot_confusion_matrix(cm, model_name, os.path.join(save_dir, "confusion_matrix.png"))
    return (fpr, tpr)


@torch.no_grad()
def evaluate_cached_head(classifier: nn.Module, model_name: str, feature_loader: DataLoader, args, device: torch.device, save_dir: Optional[str] = None, save_outputs: bool = False):
    classifier.to(device)
    classifier.eval()
    y_true_all, y_score_all, y_pred_all = [], [], []
    for features, labels in tqdm(feature_loader, desc=f"[{model_name}] Eval cached features"):
        features = features.to(device, non_blocking=True).float()
        with autocast_context(args, device):
            logits = classifier(features)
        probs = torch.softmax(logits.float(), dim=1)
        y_true_all.append(labels.detach().cpu().numpy().astype(np.int64))
        y_score_all.append(probs[:, 1].detach().cpu().numpy().astype(np.float32))
        y_pred_all.append(torch.argmax(probs, dim=1).detach().cpu().numpy().astype(np.int64))
    y_true = np.concatenate(y_true_all)
    y_score = np.concatenate(y_score_all)
    y_pred_argmax = np.concatenate(y_pred_all)
    metrics = compute_binary_metrics(y_true, y_score, y_pred_argmax, args, model_name)
    roc_xy = None
    if save_outputs:
        roc_xy = save_eval_outputs(y_true, y_score, y_pred_argmax, metrics, model_name, args, save_dir)
    else:
        roc_xy = roc_curve(y_true, y_score)[:2] if len(np.unique(y_true)) == 2 else (np.array([np.nan]), np.array([np.nan]))
    return metrics, roc_xy


@torch.no_grad()


def print_result(metrics: Dict):
    print("\n" + "=" * 80)
    print(f"[RESULT] {metrics['model']}")
    for k in ["accuracy", "balanced_accuracy", "precision", "recall_sensitivity", "specificity", "f1", "auroc", "auprc"]:
        print(f"{k}: {metrics[k]:.6f}")
    print(f"Confusion matrix [[TN, FP], [FN, TP]]: [[{metrics['tn']}, {metrics['fp']}], [{metrics['fn']}, {metrics['tp']}]]")
    print("=" * 80)


def train_cached_head(classifier: ClassificationHead, model_name: str, train_feature_loader: DataLoader, test_feature_loader: DataLoader, train_labels: np.ndarray, args, device: torch.device, save_dir: str):
    ensure_dir(save_dir)
    classifier.to(device)
    classifier.train()
    class_weight = get_class_weights_from_labels(train_labels, device) if args.class_weight == "balanced" else None
    if class_weight is not None:
        print(f"[INFO] class_weight={class_weight.detach().cpu().numpy().tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma) if args.lr_step_size > 0 else None
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.amp == "fp16"))
    history = []

    for epoch in range(1, args.max_epoch + 1):
        classifier.train()
        running_loss, correct, seen = 0.0, 0, 0
        pbar = tqdm(train_feature_loader, desc=f"[{model_name}] Train cached head {epoch}/{args.max_epoch}")
        for features, labels in pbar:
            features = features.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(args, device):
                logits = classifier(features)
                loss = criterion(logits.float(), labels)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if args.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(classifier.parameters(), args.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(classifier.parameters(), args.grad_clip_norm)
                optimizer.step()

            with torch.no_grad():
                pred = torch.argmax(logits.float(), dim=1)
                bsz = int(labels.shape[0])
                seen += bsz
                correct += int((pred == labels).sum().item())
                running_loss += float(loss.detach().cpu().item()) * bsz
            pbar.set_postfix(loss=running_loss / max(seen, 1), acc=correct / max(seen, 1))

        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(seen, 1),
            "train_acc": correct / max(seen, 1),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }

        if args.eval_every_epoch > 0 and (epoch % int(args.eval_every_epoch) == 0 or epoch == args.max_epoch):
            eval_metrics, _ = evaluate_cached_head(classifier, model_name, test_feature_loader, args, device, save_outputs=False)
            for k, v in eval_metrics.items():
                if isinstance(v, (int, float, np.integer, np.floating)):
                    row[f"eval_{k}"] = float(v)
            print(
                f"[EVAL] {model_name} epoch={epoch:03d}, "
                f"acc={eval_metrics['accuracy']:.6f}, bal_acc={eval_metrics['balanced_accuracy']:.6f}, "
                f"auroc={eval_metrics['auroc']:.6f}, auprc={eval_metrics['auprc']:.6f}"
            )

        history.append(row)
        print(f"[TRAIN] {model_name} epoch={epoch:03d}, loss={row['train_loss']:.6f}, acc={row['train_acc']:.6f}")

    pd.DataFrame(history).to_csv(os.path.join(save_dir, "train_log.csv"), index=False)
    torch.save({
        "model_name": model_name,
        "feature_dim": int(classifier.feature_dim),
        "head_type": classifier.head_type,
        "head_hidden_dim": int(classifier.hidden_dim),
        "classifier_structure": classifier.structure_string(),
        "classifier_state_dict": {k: v.detach().cpu() for k, v in classifier.state_dict().items()},
        "args": vars(args),
    }, os.path.join(save_dir, "linear_probe_last.pt"))




def plot_confusion_matrix(cm, model_name, out_path):
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    im = ax.imshow(cm)
    ax.set_title(f"{model_name} confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["non", "tumor"]); ax.set_yticklabels(["non", "tumor"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_all_rocs(roc_dict, metrics_df, output_dir):
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for model_name, (fpr, tpr) in roc_dict.items():
        if np.any(np.isnan(fpr)) or np.any(np.isnan(tpr)):
            continue
        row = metrics_df[metrics_df["model"] == model_name]
        auc_val = row["auroc"].iloc[0] if len(row) else np.nan
        ax.plot(fpr, tpr, label=f"{model_name} (AUROC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Tumor detection ROC curves")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=300)
    fig.savefig(os.path.join(output_dir, "roc_curves.pdf"))
    plt.close(fig)


def run_one_model(train_ds, test_ds, args, device, output_dir):
    encoder, feature_dim, model_name = build_encoder(args, "spavista")
    head_rng_state = capture_rng_state()
    model_dir = os.path.join(output_dir, sanitize_name(model_name))
    ensure_dir(model_dir)

    (train_feature_path, train_label_path), (test_feature_path, test_label_path) = load_or_extract_feature_cache(
        encoder=encoder,
        train_ds=train_ds,
        test_ds=test_ds,
        args=args,
        device=device,
        canonical_name=model_name,
        feature_dim=feature_dim,
    )
    train_dataset = CachedFeatureDataset(
        train_feature_path, train_label_path, load_to_ram=args.load_train_features_to_ram
    )
    test_dataset = CachedFeatureDataset(
        test_feature_path, test_label_path, load_to_ram=args.load_test_features_to_ram
    )
    train_loader = make_feature_loader(train_dataset, args, train=True)
    test_loader = make_feature_loader(test_dataset, args, train=False)

    # Keep the downstream head identical whether features are newly extracted
    # or loaded from an existing cache.
    restore_rng_state(head_rng_state)
    classifier = ClassificationHead(
        feature_dim, num_classes=2, head_type=args.head_type, hidden_dim=args.head_hidden_dim
    )
    train_labels = np.load(train_label_path, mmap_mode=None)
    train_cached_head(
        classifier, model_name, train_loader, test_loader, train_labels, args, device, model_dir
    )
    metrics, roc_xy = evaluate_cached_head(
        classifier, model_name, test_loader, args, device, model_dir, save_outputs=True
    )
    print_result(metrics)
    return metrics, roc_xy

def main():
    parser = argparse.ArgumentParser(description="Tumor detection with cached frozen SpaVista HFAM features.")
    parser.add_argument("--config", type=str, default="configs/paths.yaml", help="YAML file containing checkpoint and dataset paths.")
    parser.add_argument("--root_dir", type=str, default=None)
    parser.add_argument("--test_folders", type=str, default=None)
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--normal", type=str2bool, default=True)
    parser.add_argument("--check_patch", type=str2bool, default=True)
    parser.add_argument("--cache_images", type=str2bool, default=True)
    parser.add_argument("--project_root", type=str, default="")

    # Encoder checkpoints
    parser.add_argument("--spavista_ckpt", type=str, default=None)
    parser.add_argument("--spavista_arch", type=str, default="mamba_vision_base")

    parser.add_argument("--strict_load", type=str2bool, default=False)

    # Feature cache
    parser.add_argument("--feature_cache_dir", type=str, default=None, help="Directory for reusable cached feature .npy files.")
    parser.add_argument("--train_feature_path", type=str, default="", help="Optional explicit train feature .npy path.")
    parser.add_argument("--test_feature_path", type=str, default="", help="Optional explicit test feature .npy path.")
    parser.add_argument("--train_label_path", type=str, default="", help="Optional explicit train label .npy path.")
    parser.add_argument("--test_label_path", type=str, default="", help="Optional explicit test label .npy path.")
    parser.add_argument("--force_extract_features", type=str2bool, default=False, help="True: re-extract features even if cache exists.")
    parser.add_argument("--feature_cache_dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--feature_batch_size", type=int, default=1024, help="Image batch size used for frozen encoder feature extraction.")
    parser.add_argument("--head_batch_size", type=int, default=64, help="Feature-vector batch size used for cached-feature head training.")
    parser.add_argument("--feature_num_workers", type=int, default=0, help="Workers for cached-feature DataLoader. Keep 0 for memmap safety.")
    parser.add_argument("--load_train_features_to_ram", type=str2bool, default=True)
    parser.add_argument("--load_test_features_to_ram", type=str2bool, default=True)

    # Head and training
    parser.add_argument("--head_type", type=str, default="mlp1024", choices=["linear", "mlp1024", "two_layer", "two_layer_linear"], help="linear: single Linear. mlp1024/two_layer: Linear(dim,1024)->Linear(1024,2).")
    parser.add_argument("--head_hidden_dim", type=int, default=1024)
    parser.add_argument("--max_epoch", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr_step_size", type=int, default=50)
    parser.add_argument("--lr_gamma", type=float, default=0.9)
    parser.add_argument("--grad_clip_norm", type=float, default=0.0)
    parser.add_argument("--class_weight", type=str, default="balanced", choices=["none", "balanced"])
    parser.add_argument("--eval_every_epoch", type=int, default=15, help="Evaluate on test set every N epochs. 0 disables intermediate evaluation.")

    # Evaluation
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--use_threshold_for_metrics", type=str2bool, default=True)

    # Runtime
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--tag", type=str, default="tumor_detection_linear_probe")
    parser.add_argument("--num_workers", type=int, default=0, help="Workers for image patch loading. Use 0 when --cache_images True to avoid duplicating WSI arrays.")
    parser.add_argument("--device_num", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--amp", type=str, default="fp16", choices=["none", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()
    config = load_path_config(args.config)
    task_config = config.get("tumor_detection", {}) or {}
    args.root_dir = args.root_dir or task_config.get("root_dir")
    args.test_folders = args.test_folders or task_config.get("test_folders") or "23377,23903,23268,23508,23287"
    args.spavista_ckpt = args.spavista_ckpt or config.get("checkpoint")
    args.feature_cache_dir = args.feature_cache_dir or task_config.get("feature_cache_dir") or "data/tumor_detection/TRISTdata/feature_cache"
    if not args.root_dir:
        parser.error("Set tumor_detection.root_dir in --config or pass --root_dir.")
    if not args.spavista_ckpt:
        parser.error("Set checkpoint in --config or pass --spavista_ckpt.")

    args.head_type = canonical_head_type(args.head_type)
    seed_torch(args.seed)
    add_project_root(args.project_root)
    if args.output_dir is None or str(args.output_dir).strip() == "":
        args.output_dir = os.path.join("results", args.tag)
    ensure_dir(args.output_dir)
    save_run_config(args, args.output_dir)
    device = get_device(args)
    test_folders = parse_comma_list(args.test_folders)
    model_name = "SpaVista"

    print("=" * 80)
    print("Tumor detection: frozen encoder + classification head")
    print(f"root_dir:             {args.root_dir}")
    print(f"test_folders:         {test_folders}")
    print(f"model:                {model_name}")
    print(f"feature_cache_dir:    {args.feature_cache_dir}")
    print(f"normal:               {args.normal}")
    print(f"head_type:            {args.head_type}")
    print(f"eval_every_epoch:     {args.eval_every_epoch}")
    print(f"device:               {device}")
    print("=" * 80)

    train_ds = STNetDataClassifyLazy(
        args.root_dir,
        test_folders,
        train=True,
        patch_size=args.patch_size,
        train_aug=False,
        normal=args.normal,
        check_patch=args.check_patch,
        cache_images=args.cache_images,
        verbose=True,
    )
    test_ds = STNetDataClassifyLazy(
        args.root_dir,
        test_folders,
        train=False,
        patch_size=args.patch_size,
        train_aug=False,
        normal=args.normal,
        check_patch=args.check_patch,
        cache_images=args.cache_images,
        verbose=True,
    )

    metrics, roc_xy = run_one_model(train_ds, test_ds, args, device, args.output_dir)
    all_metrics = [metrics]
    roc_dict = {metrics["model"]: roc_xy}

    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = os.path.join(args.output_dir, "all_model_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[SAVED] {metrics_path}")
    plot_all_rocs(roc_dict, metrics_df, args.output_dir)
    print("\n" + "=" * 80)
    print("All model metrics")
    print(metrics_df)
    print("=" * 80)
    print("Finished.")


if __name__ == "__main__":
    main()
