# models/mambavision_ibot.py
# -*- coding: utf-8 -*-

"""MambaVision backbone adapter used by SpaVista downstream tasks.

The adapter exposes the hierarchical feature maps required by HFAM while
retaining the parameter structure expected by the released checkpoint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaVisionForIBOT(nn.Module):
    """
    Adapter for using MambaVision as an iBOT backbone.

    MambaVision has no native [CLS] token. For iBOT compatibility:
        - pseudo CLS token = global average pooling of final feature map
        - patch tokens     = flattened final feature map tokens

    For 224x224 input and mamba_vision_T:
        final feature map = [B, 640, 7, 7]
        output tokens     = [B, 50, 640]
                          = [1 pseudo CLS + 49 region tokens]

    Masking follows iBOT-Swin:
        - iBOT mask is generated on a coarse 7x7 grid when pred_size=32.
        - MambaVision patch_embed output is usually 56x56 for 224x224 input.
        - The 7x7 mask is expanded to 56x56.
        - Masked patch_embed features are replaced by learned masked_embed.
    """

    def __init__(
        self,
        model_name="mamba_vision_T",
        pretrained=False,
        model_path=None,
        drop_path_rate=0.1,
        return_all_tokens=True,
        masked_im_modeling=True,
        num_classes=0,
    ):
        super().__init__()

        try:
            from mambavision import create_model
        except ImportError as e:
            raise ImportError(
                f"MambaVision import/init failed. Real error: {repr(e)}"
            )

        self.model_name = model_name
        self.return_all_tokens = return_all_tokens
        self.masked_im_modeling = masked_im_modeling

        kwargs = dict(
            pretrained=pretrained,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
        )
        if model_path is not None:
            kwargs["model_path"] = model_path

        self.model = create_model(model_name, **kwargs)

        # Final feature dimension used by iBOTHead.
        dim_map = {
            "mamba_vision_T": 640,
            "mamba_vision_T2": 640,
            "mamba_vision_S": 768,
            "mamba_vision_B": 1024,
            "mamba_vision_L": 1568,
            "mamba_vision_L2": 1568,
        }
        if hasattr(self.model, "norm") and hasattr(self.model.norm, "num_features"):
            self.num_features = self.model.norm.num_features
        elif hasattr(self.model, "head") and hasattr(self.model.head, "in_features"):
            self.num_features = self.model.head.in_features
        elif model_name in dim_map:
            self.num_features = dim_map[model_name]
        else:
            raise ValueError(f"Unknown MambaVision output dimension for {model_name}")
        self.embed_dim = self.num_features

        # Channel dimension right after MambaVision patch_embed.
        patch_embed_dim = self._infer_patch_embed_dim()

        # Learned mask token in patch_embed feature space, following iBOT-Swin.
        # Shape is NCHW-friendly: [1, C_patch_embed, 1, 1].
        self.masked_embed = nn.Parameter(torch.zeros(1, patch_embed_dim, 1, 1))
        nn.init.trunc_normal_(self.masked_embed, std=0.02)

    def _infer_patch_embed_dim(self):
        """
        Infer the channel dimension of self.model.patch_embed output.

        We avoid hard-coding because different MambaVision variants use
        different base dimensions. This function first checks common module
        attributes, then falls back to a CPU dummy forward through patch_embed.
        """
        patch_embed = self.model.patch_embed

        # Common direct attributes.
        for attr in ["out_channels", "num_features", "embed_dim", "dim"]:
            if hasattr(patch_embed, attr):
                value = getattr(patch_embed, attr)
                if isinstance(value, int):
                    return value

        # Common nested conv attributes.
        for attr in ["proj", "projection", "conv", "stem"]:
            if hasattr(patch_embed, attr):
                module = getattr(patch_embed, attr)
                if hasattr(module, "out_channels"):
                    return int(module.out_channels)
                # Sometimes it is a Sequential; find the last Conv2d.
                if isinstance(module, nn.Sequential):
                    for submodule in reversed(module):
                        if isinstance(submodule, nn.Conv2d):
                            return int(submodule.out_channels)

        # Robust fallback: run patch_embed once on CPU.
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = self.model.patch_embed(dummy)
        if was_training:
            self.model.train()

        if out.dim() != 4:
            raise RuntimeError(
                f"Expected MambaVision patch_embed output to be NCHW, got shape {tuple(out.shape)}"
            )
        return int(out.shape[1])

    def no_weight_decay(self):
        """
        Optional hook for optimizers that respect no_weight_decay().
        """
        return {"masked_embed"}

    def _expand_mask_to_feature_grid(self, mask, target_hw):
        """
        Expand a coarse iBOT mask to the patch_embed feature-map size.

        mask:
            [B, h, w] or [B, 1, h, w]
        target_hw:
            (H, W) of patch_embed feature map, e.g. (56, 56)

        iBOT-Swin uses repeat_interleave when the target size is an integer
        multiple of mask size. We do the same when possible, and fall back to
        nearest-neighbor interpolation otherwise.
        """
        if mask.dim() == 4:
            if mask.shape[1] != 1:
                raise RuntimeError(f"Expected mask shape [B,1,h,w], got {tuple(mask.shape)}")
            mask = mask.squeeze(1)
        if mask.dim() != 3:
            raise RuntimeError(f"Expected mask shape [B,h,w] or [B,1,h,w], got {tuple(mask.shape)}")

        mask = mask.float()
        H, W = target_hw
        h, w = mask.shape[-2:]

        if (h, w) == (H, W):
            return mask

        if H % h == 0 and W % w == 0:
            htimes = H // h
            wtimes = W // w
            mask = mask.repeat_interleave(htimes, dim=-2).repeat_interleave(wtimes, dim=-1)
            return mask

        mask = F.interpolate(mask.unsqueeze(1), size=(H, W), mode="nearest").squeeze(1)
        return mask

    def mask_model(self, feat, mask):
        """
        Apply learned mask embedding after patch_embed, following iBOT-Swin.

        feat:
            patch_embed feature map, [B, C, H, W]
        mask:
            coarse iBOT mask, normally [B, 7, 7] for 224 input with pred_size=32

        Returns:
            masked feature map, [B, C, H, W]
        """
        if mask is None:
            return feat
        if feat.dim() != 4:
            raise RuntimeError(f"Expected patch_embed feature map [B,C,H,W], got {tuple(feat.shape)}")

        B, C, H, W = feat.shape
        mask = self._expand_mask_to_feature_grid(mask, (H, W)).to(device=feat.device)
        if mask.shape[0] != B:
            raise RuntimeError(f"Mask batch size {mask.shape[0]} does not match feature batch size {B}")

        mask = mask.to(dtype=torch.bool).unsqueeze(1)  # [B, 1, H, W]
        masked_embed = self.masked_embed.to(device=feat.device, dtype=feat.dtype)

        # torch.where broadcasts [1,C,1,1] to [B,C,H,W].
        feat = torch.where(mask, masked_embed, feat)
        return feat

    def forward_features_map(self, x, mask=None):
        """
        Return final spatial feature map [B, C, H, W].

        We intentionally avoid model.forward_features(), because the official
        implementation applies avgpool and returns [B, C]. For iBOT patch loss,
        we need the final spatial map before avgpool.
        """
        x = self.model.patch_embed(x)  # usually [B, C0, H/4, W/4]

        if self.masked_im_modeling and mask is not None:
            x = self.mask_model(x, mask)

        for level in self.model.levels:
            x = level(x)

        x = self.model.norm(x)
        return x

    def forward(self, x, mask=None):
        feat = self.forward_features_map(x, mask=mask)  # [B, C, H, W]
        B, C, H, W = feat.shape

        patch_tokens = feat.flatten(2).transpose(1, 2).contiguous()  # [B, H*W, C]
        cls_token = feat.mean(dim=(2, 3)).unsqueeze(1)               # [B, 1, C]
        tokens = torch.cat([cls_token, patch_tokens], dim=1)         # [B, 1+H*W, C]

        if self.return_all_tokens:
            return tokens
        return tokens[:, 0]


# ---- factory functions used by models.__dict__[args.arch] ----

def mamba_vision_tiny(
    pretrained=False,
    model_path=None,
    drop_path_rate=0.1,
    return_all_tokens=True,
    masked_im_modeling=True,
    **kwargs,
):
    return MambaVisionForIBOT(
        model_name="mamba_vision_T",
        pretrained=pretrained,
        model_path=model_path,
        drop_path_rate=drop_path_rate,
        return_all_tokens=return_all_tokens,
        masked_im_modeling=masked_im_modeling,
    )


def mamba_vision_t2(
    pretrained=False,
    model_path=None,
    drop_path_rate=0.1,
    return_all_tokens=True,
    masked_im_modeling=True,
    **kwargs,
):
    return MambaVisionForIBOT(
        model_name="mamba_vision_T2",
        pretrained=pretrained,
        model_path=model_path,
        drop_path_rate=drop_path_rate,
        return_all_tokens=return_all_tokens,
        masked_im_modeling=masked_im_modeling,
    )


def mamba_vision_small(
    pretrained=False,
    model_path=None,
    drop_path_rate=0.1,
    return_all_tokens=True,
    masked_im_modeling=True,
    **kwargs,
):
    return MambaVisionForIBOT(
        model_name="mamba_vision_S",
        pretrained=pretrained,
        model_path=model_path,
        drop_path_rate=drop_path_rate,
        return_all_tokens=return_all_tokens,
        masked_im_modeling=masked_im_modeling,
    )

def mamba_vision_base(
    pretrained=False,
    model_path=None,
    drop_path_rate=0.1,
    return_all_tokens=True,
    masked_im_modeling=True,
    **kwargs,
):
    return MambaVisionForIBOT(
        model_name="mamba_vision_B",
        pretrained=pretrained,
        model_path=model_path,
        drop_path_rate=drop_path_rate,
        return_all_tokens=return_all_tokens,
        masked_im_modeling=masked_im_modeling,
    )

def mamba_vision_large(
    pretrained=False,
    model_path=None,
    drop_path_rate=0.1,
    return_all_tokens=True,
    masked_im_modeling=True,
    **kwargs,
):
    return MambaVisionForIBOT(
        model_name="mamba_vision_L",
        pretrained=pretrained,
        model_path=model_path,
        drop_path_rate=drop_path_rate,
        return_all_tokens=return_all_tokens,
        masked_im_modeling=masked_im_modeling,
    )


def mamba_vision_l2(
    pretrained=False,
    model_path=None,
    drop_path_rate=0.1,
    return_all_tokens=True,
    masked_im_modeling=True,
    **kwargs,
):
    return MambaVisionForIBOT(
        model_name="mamba_vision_L2",
        pretrained=pretrained,
        model_path=model_path,
        drop_path_rate=drop_path_rate,
        return_all_tokens=return_all_tokens,
        masked_im_modeling=masked_im_modeling,
    )
