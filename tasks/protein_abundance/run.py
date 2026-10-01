#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HEMIT protein marker abundance regression with frozen SpaVista HFAM features and bounded evaluation.

Model: SpaVista with HFAM.

Task:
  H&E patch -> protein marker abundance vector

Default targets:
  panCK_mean
  panCK_pos_area
  CD3_mean
  CD3_pos_area

The released metadata CSV contains:
  split, tile_id, he_path, label_path,
  panCK_mean, panCK_pos_area, CD3_mean, CD3_pos_area, ...

Key design:
  - HEMIT H&E images are resized to 224x224 by default.
  - Frozen encoder features are extracted once without image augmentation and cached.
  - The downstream head is the same two-layer linear head by default:
        Linear(feature_dim, 1024) -> Linear(1024, target_dim)
  - Targets are used on their original scale by default; optional z-scoring
    uses train-set mean and standard deviation.
  - Metrics are computed on the original target scale.
  - Both mean and median across targets are reported. Median metrics are printed first.

Outputs:
  results/<tag>/
    run_config.json
    all_model_test_metrics.csv
    all_model_target_metrics.csv
    <MODEL>/
      best_regressor.pt
      last_regressor.pt
      target_normalization.csv
      train_log.csv
      val_predictions.csv
      test_predictions.csv
      val_metrics_overall.csv
      val_metrics_per_target.csv
      test_metrics_overall.csv
      test_metrics_per_target.csv
"""

import argparse
import hashlib
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

from spavista.config import load_path_config

try:
    import timm
except Exception as e:
    raise ImportError("This script needs timm. Install it with: pip install timm") from e

try:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
except Exception as e:
    raise ImportError("This script needs scikit-learn. Install it with: pip install scikit-learn") from e

try:
    from scipy.stats import pearsonr, spearmanr
except Exception as e:
    raise ImportError("This script needs scipy. Install it with: pip install scipy") from e


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


def resolve_hemit_path(path_value: str, data_root: str, split: str, modality: str) -> str:
    """Resolve portable metadata paths against a local HEMIT dataset root."""
    path = os.path.expanduser(str(path_value))
    if os.path.isabs(path) and os.path.exists(path):
        return path
    if not data_root:
        return path
    if not os.path.isabs(path):
        return os.path.normpath(os.path.join(data_root, path))
    return os.path.join(data_root, split, modality, os.path.basename(path))


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
    if path is not None and str(path).strip() != "":
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


def get_numpy_dtype(dtype_name: str):
    dtype_name = str(dtype_name).lower()
    if dtype_name in ["float16", "fp16", "half"]:
        return np.float16
    if dtype_name in ["float32", "fp32", "single"]:
        return np.float32
    raise ValueError("--feature_cache_dtype must be float16 or float32")


# =============================================================================
# Dataset
# =============================================================================
class HEMITAbundanceDataset(Dataset):
    """
    HEMIT H&E patch dataset for protein abundance regression.

    CSV columns required:
      split, he_path, tile_id
      target columns, default:
        panCK_mean, panCK_pos_area, CD3_mean, CD3_pos_area

    Original HEMIT input images are 1024x1024. This dataset resizes them to
    patch_size, default 224, for fair feature extraction across the 4 FMs.
    """

    def __init__(
        self,
        csv_path: str,
        split: str,
        target_cols: List[str],
        data_root: str = "",
        patch_size: int = 224,
        normal: bool = True,
        strict_path_check: bool = True,
        verbose: bool = True,
    ):
        super().__init__()
        check_file(csv_path, "abundance csv")
        self.csv_path = csv_path
        self.split = str(split)
        self.target_cols = list(target_cols)
        self.patch_size = int(patch_size)
        self.normal = bool(normal)
        self.verbose = bool(verbose)

        df = pd.read_csv(csv_path)
        if "split" not in df.columns:
            raise ValueError("CSV must contain column: split")
        if "he_path" not in df.columns:
            raise ValueError("CSV must contain column: he_path")
        if "tile_id" not in df.columns:
            df["tile_id"] = np.arange(len(df)).astype(str)

        missing_targets = [c for c in self.target_cols if c not in df.columns]
        if missing_targets:
            raise ValueError(f"CSV missing target columns: {missing_targets}")

        df = df[df["split"].astype(str) == self.split].copy()
        if len(df) == 0:
            raise RuntimeError(f"No samples found for split={split} in {csv_path}")

        df["he_path"] = df["he_path"].map(
            lambda path: resolve_hemit_path(path, data_root, self.split, "input")
        )

        for c in self.target_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=self.target_cols + ["he_path"]).copy()

        if strict_path_check:
            exists = df["he_path"].map(lambda p: os.path.exists(str(p))).values
            n_missing = int((~exists).sum())
            if n_missing > 0:
                print(f"[WARN] split={split}: missing H&E paths: {n_missing}. These samples will be removed.")
                df = df.loc[exists].copy()

        if len(df) == 0:
            raise RuntimeError(f"No valid samples left for split={split} after path/target filtering.")

        self.df = df.reset_index(drop=True)
        self.targets = self.df[self.target_cols].values.astype(np.float32)

        if self.verbose:
            print("-" * 80)
            print(f"[DATASET] HEMIT split={self.split}")
            print(f"[DATASET] samples={len(self.df):,}")
            print(f"[DATASET] patch_size={self.patch_size}, normal={self.normal}")
            print(f"[DATASET] target_cols={self.target_cols}")
            print(self.df[self.target_cols].describe())
            print("-" * 80)

    def __len__(self):
        return len(self.df)

    def transform(self, image: Image.Image) -> torch.Tensor:
        if image.size != (self.patch_size, self.patch_size):
            image = image.resize((self.patch_size, self.patch_size), resample=Image.BILINEAR)
        x = TF.to_tensor(image)
        if self.normal:
            x = TF.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return x

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        he_path = str(row["he_path"])
        image = Image.open(he_path).convert("RGB")
        image = self.transform(image)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return image, y, int(idx)


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
        self.encoder = getattr(models, arch)(
            pretrained=False,
            return_all_tokens=False,
            masked_im_modeling=False,
        )

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
        common_channels = {
            64, 80, 96, 128, 160, 192, 256, 320, 384, 512, 640,
            768, 784, 1024, 1536, 1568, 2048,
        }
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
        spatial = spatial / (spatial.mean(dim=(2, 3), keepdim=True) + eps)
        ch_mean = feat_abs.mean(dim=(2, 3), keepdim=True)
        active = (feat_abs > ch_mean).float().sum(dim=(2, 3), keepdim=True)
        channel = torch.log((float(H * W) + eps) / (active + eps))
        channel = channel / (channel.mean(dim=1, keepdim=True) + eps)
        return (feat * spatial * channel).mean(dim=(2, 3))

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
    """Build the frozen SpaVista encoder used by downstream tasks."""
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


class CachedRegressionDataset(Dataset):
    def __init__(
        self,
        feature_path: str,
        target_path: str,
        target_mean: Optional[np.ndarray] = None,
        target_std: Optional[np.ndarray] = None,
        zscore_targets: bool = True,
        load_to_ram: bool = True,
    ):
        mmap_mode = None if load_to_ram else "r"
        self.features = np.load(feature_path, mmap_mode=mmap_mode)
        self.targets_raw = np.load(target_path, mmap_mode=mmap_mode).astype(np.float32)

        if self.features.ndim != 2:
            raise ValueError(f"features should be [N,D], got {self.features.shape}")
        if self.targets_raw.ndim != 2 or self.targets_raw.shape[0] != self.features.shape[0]:
            raise ValueError(f"targets should be [N,T] and match features, got features={self.features.shape}, targets={self.targets_raw.shape}")

        self.zscore_targets = bool(zscore_targets)

        if self.zscore_targets:
            if target_mean is None or target_std is None:
                raise ValueError("target_mean and target_std are required when zscore_targets=True.")
            self.target_mean = target_mean.astype(np.float32)
            self.target_std = target_std.astype(np.float32)
            self.targets = (self.targets_raw - self.target_mean[None, :]) / (self.target_std[None, :] + 1e-8)
        else:
            self.target_mean = np.zeros((self.targets_raw.shape[1],), dtype=np.float32)
            self.target_std = np.ones((self.targets_raw.shape[1],), dtype=np.float32)
            self.targets = self.targets_raw

        self.targets = self.targets.astype(np.float32)

    def __len__(self):
        return int(self.features.shape[0])

    def __getitem__(self, idx):
        feat = torch.from_numpy(np.asarray(self.features[idx])).float()
        y = torch.from_numpy(np.asarray(self.targets[idx])).float()
        return feat, y


def make_image_loader(dataset: Dataset, args, batch_size: Optional[int] = None) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    if batch_size is None:
        batch_size = args.feature_batch_size
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=args.num_workers > 0,
    )


def make_feature_loader(dataset: Dataset, args, train: bool) -> DataLoader:
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


def _file_signature(path: str) -> Dict:
    resolved = os.path.abspath(os.path.expanduser(str(path)))
    stat = os.stat(resolved)
    return {"path": resolved, "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def feature_cache_paths(args, canonical_name: str, feature_dim: int, split_name: str):
    signature = {
        "model": canonical_name,
        "checkpoint": _file_signature(args.spavista_ckpt),
        "metadata": _file_signature(args.csv_path),
        "arch": args.spavista_arch,
        "feature_pool": "HFAM",
        "feature_dim": int(feature_dim),
        "split": split_name,
        "patch_size": int(args.patch_size),
        "normal": bool(args.normal),
    }
    digest = hashlib.sha256(
        json.dumps(signature, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    prefix = f"{sanitize_name(canonical_name)}_{split_name}_{digest}"
    ensure_dir(args.feature_cache_dir)
    return (
        os.path.join(args.feature_cache_dir, f"features_{prefix}.npy"),
        os.path.join(args.feature_cache_dir, f"targets_{prefix}.npy"),
        os.path.join(args.feature_cache_dir, f"meta_{prefix}.json"),
        signature,
    )


def find_compatible_feature_cache(args, canonical_name: str, feature_dim: int, split_name: str):
    """Find a SpaVista cache copied with the dataset, regardless of old paths."""
    prefix = f"features_{sanitize_name(canonical_name)}_{split_name}_"
    compatible = []
    for name in os.listdir(args.feature_cache_dir):
        if not (name.startswith(prefix) and name.endswith(".npy")):
            continue
        suffix = name[len("features_"):-4]
        feature_path = os.path.join(args.feature_cache_dir, name)
        target_path = os.path.join(args.feature_cache_dir, f"targets_{suffix}.npy")
        meta_path = os.path.join(args.feature_cache_dir, f"meta_{suffix}.json")
        if not (os.path.isfile(target_path) and os.path.isfile(meta_path)):
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
            saved = meta.get("signature", {})
            if (
                saved.get("model") == canonical_name
                and saved.get("split") == split_name
                and saved.get("arch") == args.spavista_arch
                and saved.get("feature_pool") == "HFAM"
                and int(saved.get("feature_dim", -1)) == int(feature_dim)
                and int(saved.get("patch_size", -1)) == int(args.patch_size)
                and bool(saved.get("normal")) == bool(args.normal)
                and saved.get("checkpoint", {}).get("size") == os.stat(args.spavista_ckpt).st_size
            ):
                compatible.append((feature_path, target_path, meta_path))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    if len(compatible) > 1:
        raise ValueError(f"Multiple compatible {split_name} SpaVista caches found: {compatible}")
    return compatible[0] if compatible else None


def validate_feature_cache(
    feature_path: str,
    target_path: str,
    expected_samples: int,
    feature_dim: int,
    target_dim: int,
    split_name: str,
    expected_targets=None,
):
    features = np.load(feature_path, mmap_mode="r")
    targets = np.load(target_path, mmap_mode="r")
    expected_feature_shape = (int(expected_samples), int(feature_dim))
    expected_target_shape = (int(expected_samples), int(target_dim))
    if tuple(features.shape) != expected_feature_shape:
        raise ValueError(
            f"Invalid {split_name} feature cache shape {features.shape}; "
            f"expected {expected_feature_shape}. Use --force_extract_features True."
        )
    if tuple(targets.shape) != expected_target_shape:
        raise ValueError(
            f"Invalid {split_name} target cache shape {targets.shape}; "
            f"expected {expected_target_shape}. Use --force_extract_features True."
        )
    if expected_targets is not None and not np.array_equal(targets, np.asarray(expected_targets)):
        raise ValueError(f"{split_name} cached targets do not match the current metadata sample order: {target_path}")
    print(f"[CACHE] Using {split_name} features: {feature_path}, shape={features.shape}")


@torch.no_grad()
def extract_features_to_cache(
    encoder: nn.Module,
    dataset: Dataset,
    args,
    device: torch.device,
    canonical_name: str,
    feature_dim: int,
    target_dim: int,
    split_name: str,
    feature_path: str,
    target_path: str,
    meta_path: str,
    signature: Dict,
):
    encoder.to(device)
    encoder.eval()
    feature_dtype = get_numpy_dtype(args.feature_cache_dtype)
    sample_count = len(dataset)
    features = np.lib.format.open_memmap(
        feature_path,
        mode="w+",
        dtype=feature_dtype,
        shape=(sample_count, feature_dim),
    )
    targets = np.lib.format.open_memmap(
        target_path,
        mode="w+",
        dtype=np.float32,
        shape=(sample_count, target_dim),
    )

    offset = 0
    loader = make_image_loader(dataset, args)
    for images, batch_targets, _ in tqdm(
        loader, desc=f"[{canonical_name}] Extract {split_name} features"
    ):
        images = images.to(device, non_blocking=True)
        with autocast_context(args, device):
            batch_features = encoder(images)
        batch_features = batch_features.detach().float().cpu().numpy()
        batch_targets = batch_targets.detach().float().cpu().numpy()
        batch_size = int(batch_features.shape[0])
        features[offset:offset + batch_size] = batch_features.astype(feature_dtype, copy=False)
        targets[offset:offset + batch_size] = batch_targets
        offset += batch_size

    if offset != sample_count:
        raise RuntimeError(f"Extracted {offset} {split_name} samples; expected {sample_count}.")
    features.flush()
    targets.flush()
    del features, targets
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "saved_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "signature": signature,
                "feature_dtype": args.feature_cache_dtype,
                "samples": sample_count,
                "target_dim": target_dim,
            },
            handle,
            indent=2,
        )
    print(f"[SAVED] {feature_path}")
    print(f"[SAVED] {target_path}")


def load_or_extract_all_splits(
    encoder: nn.Module,
    datasets: Dict[str, Dataset],
    args,
    device: torch.device,
    canonical_name: str,
    feature_dim: int,
    target_cols: List[str],
):
    paths = {}
    for split_name in ("train", "val", "test"):
        dataset = datasets[split_name]
        feature_path, target_path, meta_path, signature = feature_cache_paths(
            args, canonical_name, feature_dim, split_name
        )
        if not args.force_extract_features and not (os.path.exists(feature_path) and os.path.exists(target_path)):
            compatible = find_compatible_feature_cache(args, canonical_name, feature_dim, split_name)
            if compatible is not None:
                feature_path, target_path, meta_path = compatible
        should_extract = (
            args.force_extract_features
            or not os.path.exists(feature_path)
            or not os.path.exists(target_path)
        )
        if should_extract:
            extract_features_to_cache(
                encoder=encoder,
                dataset=dataset,
                args=args,
                device=device,
                canonical_name=canonical_name,
                feature_dim=feature_dim,
                target_dim=len(target_cols),
                split_name=split_name,
                feature_path=feature_path,
                target_path=target_path,
                meta_path=meta_path,
                signature=signature,
            )
        else:
            try:
                validate_feature_cache(
                    feature_path,
                    target_path,
                    len(dataset),
                    feature_dim,
                    len(target_cols),
                    split_name,
                    expected_targets=dataset.targets,
                )
            except ValueError as error:
                print(f"[CACHE] Incompatible {split_name} cache: {error}. Extracting SpaVista features.")
                feature_path, target_path, meta_path, signature = feature_cache_paths(
                    args, canonical_name, feature_dim, split_name
                )
                extract_features_to_cache(
                    encoder=encoder, dataset=dataset, args=args, device=device,
                    canonical_name=canonical_name, feature_dim=feature_dim,
                    target_dim=len(target_cols), split_name=split_name,
                    feature_path=feature_path, target_path=target_path,
                    meta_path=meta_path, signature=signature,
                )
        paths[split_name] = (feature_path, target_path, meta_path)
    return paths


# =============================================================================
# Regression head, training, evaluation
# =============================================================================
class RegressionHead(nn.Module):
    """
    Regression head for frozen features.

    head_type:
      - linear:  Linear(feature_dim, target_dim)
      - mlp1024: Linear(feature_dim, hidden_dim) -> Linear(hidden_dim, target_dim)

    For the released two-layer linear head, mlp1024 uses no activation.
    """

    def __init__(self, feature_dim: int, target_dim: int, head_type: str = "mlp1024", hidden_dim: int = 1024):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.target_dim = int(target_dim)
        self.head_type = canonical_head_type(head_type)
        self.hidden_dim = int(hidden_dim)

        if self.head_type == "linear":
            self.net = nn.Linear(self.feature_dim, self.target_dim)
        elif self.head_type == "mlp1024":
            self.net = nn.Sequential(
                nn.Linear(self.feature_dim, self.hidden_dim),
                nn.Linear(self.hidden_dim, self.target_dim),
            )
        else:
            raise ValueError(f"Unsupported head_type={head_type}")

    def forward(self, x):
        return self.net(x.float())

    def structure_string(self):
        if self.head_type == "linear":
            return f"Linear({self.feature_dim}, {self.target_dim})"
        return f"Linear({self.feature_dim}, {self.hidden_dim}) -> Linear({self.hidden_dim}, {self.target_dim})"


def canonical_head_type(head_type: str) -> str:
    h = str(head_type).lower().replace("-", "_")
    if h in ["linear", "single", "single_linear", "one_layer", "1layer", "1_layer"]:
        return "linear"
    if h in ["mlp1024", "two_layer", "two_layer_linear", "2layer", "2_layer", "double", "two"]:
        return "mlp1024"
    raise ValueError("--head_type must be one of: linear, mlp1024, two_layer, two_layer_linear")


def safe_pearson(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) < 2:
        return np.nan
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return np.nan
    return float(pearsonr(y_true, y_pred)[0])


def safe_spearman(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) < 2:
        return np.nan
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return np.nan
    return float(spearmanr(y_true, y_pred)[0])


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_cols: List[str], model_name: str, split_name: str):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    per_rows = []
    for j, name in enumerate(target_cols):
        yt = y_true[:, j]
        yp = y_pred[:, j]

        mse = float(mean_squared_error(yt, yp))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(yt, yp))
        medae = float(np.median(np.abs(yp - yt)))

        try:
            r2 = float(r2_score(yt, yp))
        except Exception:
            r2 = np.nan

        pcc = safe_pearson(yt, yp)
        scc = safe_spearman(yt, yp)

        per_rows.append({
            "model": model_name,
            "split": split_name,
            "target": name,
            "n": int(len(yt)),
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "median_absolute_error": medae,
            "r2": r2,
            "pcc": pcc,
            "scc": scc,
            "true_mean": float(np.mean(yt)),
            "true_median": float(np.median(yt)),
            "true_std": float(np.std(yt)),
            "pred_mean": float(np.mean(yp)),
            "pred_median": float(np.median(yp)),
            "pred_std": float(np.std(yp)),
        })

    per_df = pd.DataFrame(per_rows)

    flat_true = y_true.reshape(-1)
    flat_pred = y_pred.reshape(-1)
    mse_all = float(mean_squared_error(flat_true, flat_pred))

    overall = {
        "model": model_name,
        "split": split_name,
        "n": int(y_true.shape[0]),
        "target_dim": int(y_true.shape[1]),

        # Flattened metrics over all target values.
        "mse_all": mse_all,
        "rmse_all": float(np.sqrt(mse_all)),
        "mae_all": float(mean_absolute_error(flat_true, flat_pred)),
        "median_absolute_error_all": float(np.median(np.abs(flat_pred - flat_true))),

        # Mean across targets.
        "mean_target_mse": float(per_df["mse"].mean()),
        "mean_target_rmse": float(per_df["rmse"].mean()),
        "mean_target_mae": float(per_df["mae"].mean()),
        "mean_target_median_absolute_error": float(per_df["median_absolute_error"].mean()),
        "mean_target_r2": float(per_df["r2"].mean()),
        "mean_target_pcc": float(per_df["pcc"].mean()),
        "mean_target_scc": float(per_df["scc"].mean()),

        # Median across targets. These are usually more robust and are printed first.
        "median_target_mse": float(per_df["mse"].median()),
        "median_target_rmse": float(per_df["rmse"].median()),
        "median_target_mae": float(per_df["mae"].median()),
        "median_target_median_absolute_error": float(per_df["median_absolute_error"].median()),
        "median_target_r2": float(per_df["r2"].median()),
        "median_target_pcc": float(per_df["pcc"].median()),
        "median_target_scc": float(per_df["scc"].median()),
    }

    return overall, per_df


@torch.no_grad()
def predict_cached_head(model: nn.Module, loader: DataLoader, args, device):
    model.to(device)
    model.eval()

    preds = []
    trues_norm = []

    for features, targets in tqdm(loader, desc="Predict"):
        features = features.to(device, non_blocking=True).float()

        with autocast_context(args, device):
            pred = model(features)

        preds.append(pred.detach().float().cpu().numpy())
        trues_norm.append(targets.detach().float().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues_norm = np.concatenate(trues_norm, axis=0)
    return trues_norm, preds


def inverse_targets(y_norm: np.ndarray, target_mean: np.ndarray, target_std: np.ndarray, zscore: bool):
    if not zscore:
        return y_norm.astype(np.float32)
    return (y_norm * (target_std[None, :] + 1e-8) + target_mean[None, :]).astype(np.float32)


def project_predictions_to_range(
    y_pred_raw: np.ndarray,
    lower: float = 0.0,
    upper: float = 1.0,
) -> np.ndarray:
    """
    Project predictions to the physically valid target range.

    This is applied only for validation/test reporting, after predictions are
    represented on the original abundance scale. It does not affect the
    regression-head training objective.
    """
    y_pred_raw = np.asarray(y_pred_raw, dtype=np.float32)
    return np.clip(y_pred_raw, float(lower), float(upper)).astype(np.float32)


def compute_projection_qc(
    y_pred_raw: np.ndarray,
    target_cols: List[str],
    model_name: str,
    split_name: str,
    lower: float = 0.0,
    upper: float = 1.0,
) -> pd.DataFrame:
    """Summarize how often unconstrained predictions fall outside [lower, upper]."""
    y_pred_raw = np.asarray(y_pred_raw, dtype=np.float32)
    rows = []
    for j, name in enumerate(target_cols):
        yp = y_pred_raw[:, j]
        below = yp < lower
        above = yp > upper
        outside = below | above
        rows.append({
            "model": model_name,
            "split": split_name,
            "target": name,
            "n": int(len(yp)),
            "raw_min": float(np.min(yp)),
            "raw_max": float(np.max(yp)),
            "raw_mean": float(np.mean(yp)),
            "n_below_lower": int(below.sum()),
            "pct_below_lower": float(100.0 * below.mean()),
            "n_above_upper": int(above.sum()),
            "pct_above_upper": float(100.0 * above.mean()),
            "n_outside_range": int(outside.sum()),
            "pct_outside_range": float(100.0 * outside.mean()),
        })
    return pd.DataFrame(rows)


def save_predictions(
    out_path: str,
    y_true_raw: np.ndarray,
    y_pred_raw: np.ndarray,
    y_pred_bounded: np.ndarray,
    target_cols: List[str],
):
    """
    Save both raw unconstrained predictions and [0,1]-projected predictions.

    `pred_<target>` is the bounded prediction used for the primary metrics.
    `pred_raw_<target>` is retained for transparency and QC.
    """
    data = {}
    for j, name in enumerate(target_cols):
        data[f"true_{name}"] = y_true_raw[:, j]
        data[f"pred_raw_{name}"] = y_pred_raw[:, j]
        data[f"pred_{name}"] = y_pred_bounded[:, j]
        data[f"error_raw_{name}"] = y_pred_raw[:, j] - y_true_raw[:, j]
        data[f"error_{name}"] = y_pred_bounded[:, j] - y_true_raw[:, j]
    pd.DataFrame(data).to_csv(out_path, index=False)


def evaluate_cached_head(
    model: nn.Module,
    loader: DataLoader,
    args,
    device,
    target_cols: List[str],
    target_mean: np.ndarray,
    target_std: np.ndarray,
    model_name: str,
    split_name: str,
    save_dir: Optional[str] = None,
):
    y_true_norm, y_pred_norm = predict_cached_head(model, loader, args, device)

    # Return both ground truth and predictions to their original [0,1] abundance scale.
    y_true_raw = inverse_targets(y_true_norm, target_mean, target_std, args.target_zscore)
    y_pred_raw = inverse_targets(y_pred_norm, target_mean, target_std, args.target_zscore)

    # Primary reported predictions are projected to the physically valid range.
    if args.clip_predictions:
        y_pred_eval = project_predictions_to_range(
            y_pred_raw,
            lower=args.prediction_lower_bound,
            upper=args.prediction_upper_bound,
        )
    else:
        y_pred_eval = y_pred_raw.astype(np.float32)

    # Main metrics: bounded predictions when --clip_predictions=True.
    overall, per_df = compute_regression_metrics(
        y_true_raw, y_pred_eval, target_cols, model_name, split_name
    )
    overall["prediction_projection"] = bool(args.clip_predictions)
    overall["prediction_lower_bound"] = float(args.prediction_lower_bound)
    overall["prediction_upper_bound"] = float(args.prediction_upper_bound)

    # Raw metrics are saved separately so the effect of the projection is fully auditable.
    raw_overall, raw_per_df = compute_regression_metrics(
        y_true_raw, y_pred_raw, target_cols, model_name, f"{split_name}_raw_unbounded"
    )
    # Expose raw metrics in the returned dict so checkpoint selection can exactly
    # follow the original unconstrained regression protocol while final reported
    # metrics use the bounded predictions.
    for k, v in raw_overall.items():
        if isinstance(v, (int, float, np.integer, np.floating)):
            overall[f"raw_{k}"] = float(v)

    projection_qc = compute_projection_qc(
        y_pred_raw,
        target_cols,
        model_name,
        split_name,
        lower=args.prediction_lower_bound,
        upper=args.prediction_upper_bound,
    )

    if save_dir is not None:
        ensure_dir(save_dir)
        save_predictions(
            os.path.join(save_dir, f"{split_name}_predictions.csv"),
            y_true_raw,
            y_pred_raw,
            y_pred_eval,
            target_cols,
        )
        pd.DataFrame([overall]).to_csv(
            os.path.join(save_dir, f"{split_name}_metrics_overall.csv"), index=False
        )
        per_df.to_csv(
            os.path.join(save_dir, f"{split_name}_metrics_per_target.csv"), index=False
        )
        pd.DataFrame([raw_overall]).to_csv(
            os.path.join(save_dir, f"{split_name}_metrics_overall_raw_unbounded.csv"), index=False
        )
        raw_per_df.to_csv(
            os.path.join(save_dir, f"{split_name}_metrics_per_target_raw_unbounded.csv"), index=False
        )
        projection_qc.to_csv(
            os.path.join(save_dir, f"{split_name}_prediction_range_qc.csv"), index=False
        )

    return overall, per_df


def train_regression_head(
    model: RegressionHead,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args,
    device,
    target_cols,
    target_mean,
    target_std,
    save_dir: str,
):
    ensure_dir(save_dir)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma) if args.lr_step_size > 0 else None

    if args.loss == "mse":
        criterion = nn.MSELoss()
    elif args.loss == "smoothl1":
        criterion = nn.SmoothL1Loss(beta=args.smoothl1_beta)
    else:
        raise ValueError("--loss must be mse or smoothl1")

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.amp == "fp16"))

    best_score = -1e18
    best_epoch = -1
    history = []

    for epoch in range(1, args.max_epoch + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        pbar = tqdm(train_loader, desc=f"[{model_name}] Train regression head {epoch}/{args.max_epoch}")
        for features, targets in pbar:
            features = features.to(device, non_blocking=True).float()
            targets = targets.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)

            with autocast_context(args, device):
                pred = model(features)
                loss = criterion(pred.float(), targets.float())

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if args.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                optimizer.step()

            bsz = int(features.shape[0])
            running_loss += float(loss.detach().cpu().item()) * bsz
            seen += bsz
            pbar.set_postfix(loss=running_loss / max(seen, 1))

        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(seen, 1),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }

        if args.eval_every_epoch > 0 and (epoch % int(args.eval_every_epoch) == 0 or epoch == args.max_epoch):
            val_overall, _ = evaluate_cached_head(
                model=model,
                loader=val_loader,
                args=args,
                device=device,
                target_cols=target_cols,
                target_mean=target_mean,
                target_std=target_std,
                model_name=model_name,
                split_name="val",
                save_dir=None,
            )

            for k, v in val_overall.items():
                if isinstance(v, (int, float, np.integer, np.floating)):
                    row[f"val_{k}"] = float(v)

            # By default, checkpoint selection uses the same [0,1]-projected
            # validation predictions used for the primary reported metrics.
            # Raw-prediction selection is retained only as an optional sensitivity setting.
            metric_prefix = "raw_" if args.checkpoint_on_raw_predictions else ""

            if args.best_metric == "val_median_pcc":
                score = float(val_overall[f"{metric_prefix}median_target_pcc"])
            elif args.best_metric == "val_median_scc":
                score = float(val_overall[f"{metric_prefix}median_target_scc"])
            elif args.best_metric == "val_neg_median_mse":
                score = -float(val_overall[f"{metric_prefix}median_target_mse"])
            elif args.best_metric == "val_neg_median_mae":
                score = -float(val_overall[f"{metric_prefix}median_target_mae"])
            elif args.best_metric == "val_mean_pcc":
                score = float(val_overall[f"{metric_prefix}mean_target_pcc"])
            elif args.best_metric == "val_mean_scc":
                score = float(val_overall[f"{metric_prefix}mean_target_scc"])
            elif args.best_metric == "val_neg_mse":
                score = -float(val_overall[f"{metric_prefix}mean_target_mse"])
            elif args.best_metric == "val_neg_mae":
                score = -float(val_overall[f"{metric_prefix}mean_target_mae"])
            else:
                raise ValueError(f"Unknown best_metric={args.best_metric}")

            print(
                f"[VAL] {model_name} epoch={epoch:03d}, "
                f"bounded_median_pcc={val_overall['median_target_pcc']:.6f}, "
                f"raw_median_pcc={val_overall['raw_median_target_pcc']:.6f}, "
                f"bounded_median_scc={val_overall['median_target_scc']:.6f}, "
                f"raw_median_scc={val_overall['raw_median_target_scc']:.6f}, "
                f"selection={'raw' if args.checkpoint_on_raw_predictions else 'bounded'}, "
                f"score={score:.6f}"
            )

            if score > best_score:
                best_score = score
                best_epoch = epoch
                torch.save({
                    "model_name": model_name,
                    "feature_dim": int(model.feature_dim),
                    "target_dim": int(model.target_dim),
                    "target_cols": target_cols,
                    "target_zscore": bool(args.target_zscore),
                    "target_mean": target_mean.astype(np.float32),
                    "target_std": target_std.astype(np.float32),
                    "head_type": model.head_type,
                    "head_hidden_dim": int(model.hidden_dim),
                    "head_structure": model.structure_string(),
                    "best_epoch": int(best_epoch),
                    "best_score": float(best_score),
                    "best_metric": args.best_metric,
                    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "args": vars(args),
                }, os.path.join(save_dir, "best_regressor.pt"))
                print(f"[SAVE] New best model saved at epoch={epoch}, score={score:.6f}")

        history.append(row)
        print(f"[TRAIN] {model_name} epoch={epoch:03d}, loss={row['train_loss']:.6f}")

    torch.save({
        "model_name": model_name,
        "feature_dim": int(model.feature_dim),
        "target_dim": int(model.target_dim),
        "target_cols": target_cols,
        "target_zscore": bool(args.target_zscore),
        "target_mean": target_mean.astype(np.float32),
        "target_std": target_std.astype(np.float32),
        "head_type": model.head_type,
        "head_hidden_dim": int(model.hidden_dim),
        "head_structure": model.structure_string(),
        "last_epoch": int(args.max_epoch),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "args": vars(args),
    }, os.path.join(save_dir, "last_regressor.pt"))

    pd.DataFrame(history).to_csv(os.path.join(save_dir, "train_log.csv"), index=False)

    return best_epoch, best_score


def load_best_or_last(model: RegressionHead, save_dir: str, device: torch.device, prefer_best: bool = True):
    ckpt_path = os.path.join(save_dir, "best_regressor.pt") if prefer_best else os.path.join(save_dir, "last_regressor.pt")
    if prefer_best and not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(save_dir, "last_regressor.pt")
    check_file(ckpt_path, "regressor checkpoint")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device)
    model.eval()
    print(f"[LOAD] Loaded regressor checkpoint: {ckpt_path}")
    return ckpt_path


def print_overall_result(metrics: Dict):
    print("\n" + "=" * 80)
    print(f"[RESULT] {metrics['model']} split={metrics['split']}")

    print("[Median across targets]")
    for k in [
        "median_target_pcc",
        "median_target_scc",
        "median_target_mse",
        "median_target_rmse",
        "median_target_mae",
        "median_target_r2",
    ]:
        print(f"{k}: {metrics[k]:.6f}")

    print("\n[Mean across targets]")
    for k in [
        "mean_target_pcc",
        "mean_target_scc",
        "mean_target_mse",
        "mean_target_rmse",
        "mean_target_mae",
        "mean_target_r2",
    ]:
        print(f"{k}: {metrics[k]:.6f}")

    print("=" * 80)


# =============================================================================
# Main run
# =============================================================================
def run_one_model(model_key, datasets, args, device, output_dir, target_cols):
    encoder, feature_dim, canonical_name = build_encoder(args, model_key)
    model_dir = os.path.join(output_dir, sanitize_name(canonical_name))
    ensure_dir(model_dir)

    print("=" * 80)
    print(f"[MODEL] {canonical_name}")
    print(f"[MODEL] feature_dim={feature_dim}")
    print(f"[MODEL] target_dim={len(target_cols)}")
    print(f"[MODEL] target_cols={target_cols}")
    print(f"[MODEL] feature cache dir={args.feature_cache_dir}")
    print(f"[MODEL] head_type={canonical_head_type(args.head_type)}")
    print("=" * 80)

    paths = load_or_extract_all_splits(
        encoder=encoder,
        datasets=datasets,
        args=args,
        device=device,
        canonical_name=canonical_name,
        feature_dim=feature_dim,
        target_cols=target_cols,
    )

    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_feature_path, train_target_path, _ = paths["train"]
    val_feature_path, val_target_path, _ = paths["val"]
    test_feature_path, test_target_path, _ = paths["test"]

    train_targets_raw = np.load(train_target_path).astype(np.float32)
    target_mean = train_targets_raw.mean(axis=0).astype(np.float32)
    target_std = train_targets_raw.std(axis=0).astype(np.float32)
    target_std[target_std < 1e-8] = 1.0

    pd.DataFrame({
        "target": target_cols,
        "target_mean_train": target_mean,
        "target_std_train": target_std,
    }).to_csv(os.path.join(model_dir, "target_statistics.csv"), index=False)

    train_feature_ds = CachedRegressionDataset(
        train_feature_path,
        train_target_path,
        target_mean=target_mean,
        target_std=target_std,
        zscore_targets=args.target_zscore,
        load_to_ram=args.load_features_to_ram,
    )
    val_feature_ds = CachedRegressionDataset(
        val_feature_path,
        val_target_path,
        target_mean=target_mean,
        target_std=target_std,
        zscore_targets=args.target_zscore,
        load_to_ram=args.load_features_to_ram,
    )
    test_feature_ds = CachedRegressionDataset(
        test_feature_path,
        test_target_path,
        target_mean=target_mean,
        target_std=target_std,
        zscore_targets=args.target_zscore,
        load_to_ram=args.load_features_to_ram,
    )

    train_loader = make_feature_loader(train_feature_ds, args, train=True)
    val_loader = make_feature_loader(val_feature_ds, args, train=False)
    test_loader = make_feature_loader(test_feature_ds, args, train=False)

    # Reset the RNG before creating every downstream head so that head
    # initialization is independent of model order / encoder construction.
    seed_torch(args.seed)

    regressor = RegressionHead(
        feature_dim=feature_dim,
        target_dim=len(target_cols),
        head_type=args.head_type,
        hidden_dim=args.head_hidden_dim,
    )

    print(f"[HEAD] {regressor.structure_string()}")

    best_epoch, best_score = train_regression_head(
        model=regressor,
        model_name=canonical_name,
        train_loader=train_loader,
        val_loader=val_loader,
        args=args,
        device=device,
        target_cols=target_cols,
        target_mean=target_mean,
        target_std=target_std,
        save_dir=model_dir,
    )

    load_best_or_last(regressor, model_dir, device, prefer_best=True)

    val_overall, val_per_df = evaluate_cached_head(
        model=regressor,
        loader=val_loader,
        args=args,
        device=device,
        target_cols=target_cols,
        target_mean=target_mean,
        target_std=target_std,
        model_name=canonical_name,
        split_name="val",
        save_dir=model_dir,
    )

    test_overall, test_per_df = evaluate_cached_head(
        model=regressor,
        loader=test_loader,
        args=args,
        device=device,
        target_cols=target_cols,
        target_mean=target_mean,
        target_std=target_std,
        model_name=canonical_name,
        split_name="test",
        save_dir=model_dir,
    )

    test_overall["best_epoch"] = int(best_epoch)
    test_overall["best_score"] = float(best_score)
    test_overall["best_metric"] = args.best_metric
    test_overall["head_type"] = canonical_head_type(args.head_type)
    test_overall["head_hidden_dim"] = int(args.head_hidden_dim)
    test_overall["target_zscore"] = bool(args.target_zscore)
    test_overall["clip_predictions"] = bool(args.clip_predictions)
    test_overall["checkpoint_on_raw_predictions"] = bool(args.checkpoint_on_raw_predictions)
    test_overall["prediction_lower_bound"] = float(args.prediction_lower_bound)
    test_overall["prediction_upper_bound"] = float(args.prediction_upper_bound)

    pd.DataFrame([test_overall]).to_csv(os.path.join(model_dir, "test_metrics_overall.csv"), index=False)
    test_per_df.to_csv(os.path.join(model_dir, "test_metrics_per_target.csv"), index=False)

    print_overall_result(test_overall)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return test_overall, test_per_df


def main():
    parser = argparse.ArgumentParser(
        description="HEMIT abundance regression with frozen SpaVista HFAM features."
    )

    # Dataset
    parser.add_argument("--config", type=str, default="configs/paths.yaml", help="YAML file containing checkpoint and dataset paths.")
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--data_root", type=str, default="", help="Root containing HEMIT train/val/test directories.")
    parser.add_argument("--target_cols", type=str, default="panCK_mean,panCK_pos_area,CD3_mean,CD3_pos_area")
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--normal", type=str2bool, default=True)
    parser.add_argument("--strict_path_check", type=str2bool, default=True)

    # Frozen SpaVista encoder
    parser.add_argument("--project_root", type=str, default="")
    parser.add_argument("--spavista_ckpt", type=str, default=None)
    parser.add_argument("--spavista_arch", type=str, default="mamba_vision_base")
    parser.add_argument("--strict_load", type=str2bool, default=False)

    # Feature cache
    parser.add_argument("--feature_cache_dir", type=str, default=None)
    parser.add_argument("--force_extract_features", type=str2bool, default=False)
    parser.add_argument("--feature_cache_dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--feature_batch_size", type=int, default=64)
    parser.add_argument("--head_batch_size", type=int, default=64)
    parser.add_argument("--feature_num_workers", type=int, default=0)
    parser.add_argument("--load_features_to_ram", type=str2bool, default=True)

    # Downstream head
    parser.add_argument("--head_type", type=str, default="mlp1024", choices=["linear", "mlp1024", "two_layer", "two_layer_linear"])
    parser.add_argument("--head_hidden_dim", type=int, default=1024)
    parser.add_argument("--target_zscore", type=str2bool, default=False)
    parser.add_argument(
        "--clip_predictions", type=str2bool, default=True,
        help="Project validation/test predictions to the valid target range before primary metrics.",
    )
    parser.add_argument(
        "--checkpoint_on_raw_predictions", type=str2bool, default=False,
        help="Use raw rather than bounded validation predictions for best-checkpoint selection.",
    )
    parser.add_argument("--prediction_lower_bound", type=float, default=0.0)
    parser.add_argument("--prediction_upper_bound", type=float, default=1.0)
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "smoothl1"])
    parser.add_argument("--smoothl1_beta", type=float, default=1.0)
    parser.add_argument("--max_epoch", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr_step_size", type=int, default=50)
    parser.add_argument("--lr_gamma", type=float, default=0.9)
    parser.add_argument("--grad_clip_norm", type=float, default=0.0)
    parser.add_argument("--eval_every_epoch", type=int, default=1)
    parser.add_argument("--best_metric", type=str, default="val_median_pcc", choices=["val_median_pcc", "val_median_scc", "val_neg_median_mse", "val_neg_median_mae", "val_mean_pcc", "val_mean_scc", "val_neg_mse", "val_neg_mae"])

    # Runtime
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--tag", type=str, default="hemit_abundance_regression_spavista_bounded")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device_num", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--amp", type=str, default="fp16", choices=["none", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=2024)

    args = parser.parse_args()
    config = load_path_config(args.config)
    task_config = config.get("hemit", {}) or {}
    args.csv_path = args.csv_path or task_config.get("labels_csv")
    args.data_root = args.data_root or task_config.get("root") or ""
    args.spavista_ckpt = args.spavista_ckpt or config.get("checkpoint")
    args.feature_cache_dir = args.feature_cache_dir or task_config.get("feature_cache_dir") or "data/protein/HEMIT/feature_cache"
    if not args.csv_path:
        parser.error("Set hemit.labels_csv in --config or pass --csv_path.")
    if not args.data_root:
        parser.error("Set hemit.root in --config or pass --data_root.")
    if not args.spavista_ckpt:
        parser.error("Set checkpoint in --config or pass --spavista_ckpt.")
    if args.prediction_lower_bound >= args.prediction_upper_bound:
        raise ValueError("--prediction_lower_bound must be smaller than --prediction_upper_bound")

    args.head_type = canonical_head_type(args.head_type)
    seed_torch(args.seed)
    add_project_root(args.project_root)

    if args.output_dir is None or str(args.output_dir).strip() == "":
        args.output_dir = os.path.join("results", args.tag)
    ensure_dir(args.output_dir)
    save_run_config(args, args.output_dir)

    device = get_device(args)
    target_cols = parse_comma_list(args.target_cols)
    model_keys = ["spavista"]

    print("=" * 80)
    print("HEMIT abundance regression: frozen encoder + two-layer linear head")
    print(f"csv_path:             {args.csv_path}")
    print(f"target_cols:          {target_cols}")
    print(f"model:                {model_keys[0]}")
    print(f"patch_size:           {args.patch_size}")
    print(f"normal:               {args.normal}")
    print(f"feature_cache_dir:    {args.feature_cache_dir}")
    print(f"force_extract:        {args.force_extract_features}")
    print(f"head_type:            {args.head_type}")
    print(f"target_zscore:        {args.target_zscore}")
    print(f"clip_predictions:     {args.clip_predictions}")
    print(f"checkpoint_on_raw:    {args.checkpoint_on_raw_predictions}")
    print(f"prediction_range:     [{args.prediction_lower_bound}, {args.prediction_upper_bound}]")
    print(f"loss:                 {args.loss}")
    print(f"best_metric:          {args.best_metric}")
    print(f"device:               {device}")
    print("=" * 80)

    datasets = {
        "train": HEMITAbundanceDataset(
            args.csv_path,
            "train",
            target_cols=target_cols,
            data_root=args.data_root,
            patch_size=args.patch_size,
            normal=args.normal,
            strict_path_check=args.strict_path_check,
            verbose=True,
        ),
        "val": HEMITAbundanceDataset(
            args.csv_path,
            "val",
            target_cols=target_cols,
            data_root=args.data_root,
            patch_size=args.patch_size,
            normal=args.normal,
            strict_path_check=args.strict_path_check,
            verbose=True,
        ),
        "test": HEMITAbundanceDataset(
            args.csv_path,
            "test",
            target_cols=target_cols,
            data_root=args.data_root,
            patch_size=args.patch_size,
            normal=args.normal,
            strict_path_check=args.strict_path_check,
            verbose=True,
        ),
    }

    all_overall = []
    all_per_target = []

    for model_key in model_keys:
        overall, per_df = run_one_model(
            model_key=model_key,
            datasets=datasets,
            args=args,
            device=device,
            output_dir=args.output_dir,
            target_cols=target_cols,
        )
        all_overall.append(overall)
        all_per_target.append(per_df)

    overall_df = pd.DataFrame(all_overall)
    per_target_df = pd.concat(all_per_target, axis=0, ignore_index=True)

    overall_path = os.path.join(args.output_dir, "all_model_test_metrics.csv")
    per_target_path = os.path.join(args.output_dir, "all_model_target_metrics.csv")

    overall_df.to_csv(overall_path, index=False)
    per_target_df.to_csv(per_target_path, index=False)

    print("\n" + "=" * 80)
    print("[DONE] Saved summary:")
    print(overall_path)
    print(per_target_path)
    print("\nOverall metrics, median first:")

    show_cols = [
        "model",
        "median_target_pcc",
        "median_target_scc",
        "median_target_mse",
        "median_target_mae",
        "median_target_r2",
        "mean_target_pcc",
        "mean_target_scc",
        "mean_target_mse",
        "mean_target_mae",
        "mean_target_r2",
        "best_epoch",
    ]
    show_cols = [c for c in show_cols if c in overall_df.columns]
    print(overall_df[show_cols].to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
