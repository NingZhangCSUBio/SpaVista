"""Public SpaVista feature extraction API."""

from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from . import models


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _checkpoint_state_dict(checkpoint, model_keys):
    if not isinstance(checkpoint, dict):
        raise ValueError("The checkpoint must contain a state dictionary.")
    for key in ("state_dict", "model", "teacher", "student", "module", "backbone", "encoder"):
        if isinstance(checkpoint.get(key), dict):
            checkpoint = checkpoint[key]
            break

    prefixes = (
        "module.backbone.", "module.encoder.", "module.", "backbone.",
        "encoder.", "student.", "teacher.", "model.",
        "ibot_mamba.encoder.", "ibot_mamba.",
    )
    stripped = {}
    for key, value in checkpoint.items():
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
                    break
        stripped[key] = value
    raw_overlap = len(set(checkpoint).intersection(model_keys))
    stripped_overlap = len(set(stripped).intersection(model_keys))
    return stripped if stripped_overlap > raw_overlap else checkpoint


class SpaVistaFeatureExtractor(nn.Module):
    """Frozen SpaVista encoder with the released HFAM feature aggregation module."""

    def __init__(self, checkpoint_path: Union[str, Path], arch: str = "mamba_vision_base", strict: bool = False):
        super().__init__()
        if not hasattr(models, arch):
            raise ValueError(f"Unknown SpaVista architecture: {arch}")
        self.encoder = getattr(models, arch)(
            pretrained=False,
            return_all_tokens=False,
            masked_im_modeling=False,
        )
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        state_dict = _checkpoint_state_dict(checkpoint, set(self.encoder.state_dict()))
        message = self.encoder.load_state_dict(state_dict, strict=strict)
        if not strict and (message.missing_keys or message.unexpected_keys):
            print(f"Checkpoint loaded with missing keys={message.missing_keys} and unexpected keys={message.unexpected_keys}")
        self.requires_grad_(False)
        self.eval()

    @staticmethod
    def preprocess(image: Union[str, Path, Image.Image], image_size: int = 224) -> torch.Tensor:
        """Convert an RGB image to a batched, ImageNet-normalized tensor."""
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = image.convert("RGB").resize((image_size, image_size), Image.BICUBIC)
        tensor = TF.to_tensor(image)
        tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        return tensor.unsqueeze(0)

    @staticmethod
    def _to_bchw(feature):
        if feature.ndim != 4:
            raise RuntimeError(f"Expected a 4D feature map, got {tuple(feature.shape)}")
        if feature.shape[1] < feature.shape[-1] and feature.shape[1] == feature.shape[2]:
            feature = feature.permute(0, 3, 1, 2).contiguous()
        return feature

    def _late_stage_maps(self, images):
        model = self.encoder.model if hasattr(self.encoder, "model") else self.encoder
        features = []
        x = model.patch_embed(images)
        for level in model.levels:
            x = level(x)
            if isinstance(x, (tuple, list)):
                x = x[0]
            features.append(self._to_bchw(x))
        if len(features) < 4:
            raise RuntimeError(f"SpaVista returned {len(features)} stages; HFAM requires four.")
        stage3, stage4 = features[2], features[3]
        if getattr(model, "norm", None) is not None:
            try:
                stage4 = model.norm(stage4)
            except (RuntimeError, TypeError):
                pass
        return self._to_bchw(stage3), self._to_bchw(stage4)

    @staticmethod
    def _normalize(feature):
        return F.normalize(feature, p=2, dim=1, eps=1e-12)

    def _power(self, feature):
        feature = F.adaptive_avg_pool2d(feature, 1).flatten(1)
        feature = torch.sign(feature) * torch.sqrt(torch.clamp(torch.abs(feature), min=1e-12))
        return self._normalize(feature)

    def _crow(self, feature):
        absolute = torch.abs(feature)
        spatial = absolute.sum(dim=1, keepdim=True)
        spatial = spatial / (spatial.mean(dim=(2, 3), keepdim=True) + 1e-6)
        channel_mean = absolute.mean(dim=(2, 3), keepdim=True)
        active = (absolute > channel_mean).float().sum(dim=(2, 3), keepdim=True)
        channel = torch.log((float(feature.shape[2] * feature.shape[3]) + 1e-6) / (active + 1e-6))
        channel = channel / (channel.mean(dim=1, keepdim=True) + 1e-6)
        return self._normalize((feature * spatial * channel).mean(dim=(2, 3)))

    def forward(self, images):
        stage3, stage4 = self._late_stage_maps(images)
        power = torch.cat((self._power(stage3), self._power(stage4)), dim=1)
        crow = torch.cat((self._crow(stage3), self._crow(stage4)), dim=1)
        return 0.5 * (power + crow)

    @torch.inference_mode()
    def extract(self, image: Union[str, Path, Image.Image], image_size: int = 224, device: Union[str, torch.device] = "cpu") -> np.ndarray:
        """Return one HFAM feature vector as a float32 NumPy array."""
        device = torch.device(device)
        self.to(device).eval()
        feature = self(self.preprocess(image, image_size).to(device))
        return feature[0].float().cpu().numpy()
