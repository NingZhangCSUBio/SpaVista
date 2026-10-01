# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

from argparse import ArgumentParser

from spavista import models


class IBOTMambaEncoder(nn.Module):
    """
    iBOT-MambaVision encoder wrapper.

    Released feature aggregation module: HFAM.
    """

    VALID_POOLS = ["HFAM"]

    def __init__(
        self,
        checkpoint_path=None,
        arch="mamba_vision_tiny",
        strict=True,
        return_all_tokens=False,
        masked_im_modeling=False,
        feature_pool="HFAM",
    ):
        super().__init__()

        if not hasattr(models, arch):
            raise ValueError(
                f"Unknown MambaVision arch: {arch}. "
                f"Please check models/__init__.py and models/mambavision_ibot.py."
            )

        if feature_pool not in self.VALID_POOLS:
            raise ValueError(
                f"Unknown feature_pool={feature_pool}. "
                f"Choose from: {self.VALID_POOLS}."
            )

        self.arch = arch
        self.feature_pool = feature_pool

        # HFAM accesses the hierarchical MambaVision feature maps directly.
        self.encoder = getattr(models, arch)(
            pretrained=False,
            return_all_tokens=False,
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

        if checkpoint_path is not None:
            self.load_ibot_mamba_weights(checkpoint_path, strict=strict)

    def load_ibot_mamba_weights(self, checkpoint_path, strict=True):
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif isinstance(ckpt, dict):
            state_dict = ckpt
        else:
            raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

        new_state_dict = {}
        for k, v in state_dict.items():
            nk = k

            if nk.startswith("module.backbone."):
                nk = nk[len("module.backbone."):]
            elif nk.startswith("backbone."):
                nk = nk[len("backbone."):]
            elif nk.startswith("module."):
                nk = nk[len("module."):]

            new_state_dict[nk] = v

        msg = self.encoder.load_state_dict(new_state_dict, strict=strict)

        print(f"[IBOTMambaEncoder] arch: {self.arch}")
        print(f"[IBOTMambaEncoder] feature_pool: {self.feature_pool}")
        print(f"[IBOTMambaEncoder] encoder_dim: {self.encoder_dim}")
        print(f"[IBOTMambaEncoder] output_dim: {self.output_dim}")
        print(f"[IBOTMambaEncoder] Loaded checkpoint: {checkpoint_path}")
        print(f"[IBOTMambaEncoder] missing keys: {msg.missing_keys}")
        print(f"[IBOTMambaEncoder] unexpected keys: {msg.unexpected_keys}")

    @staticmethod
    def _to_bchw(x):
        """
        Convert feature map to [B, C, H, W] if needed.
        MambaVision normally returns [B, C, H, W]; retain compatibility with channel-last maps.
        """
        if x.dim() != 4:
            raise RuntimeError(f"Expected 4D feature map, got shape={tuple(x.shape)}")

        # Convert [B, H, W, C] to [B, C, H, W] when needed.
        common_channels = {
            64, 80, 96, 128, 160, 192, 256, 320, 384,
            512, 640, 768, 784, 1024, 1568, 2048
        }
        if x.shape[1] not in common_channels and x.shape[-1] in common_channels:
            x = x.permute(0, 3, 1, 2).contiguous()

        return x

    def _get_mambavision_model(self):
        """
        For a MambaVisionForIBOT adapter, the underlying MambaVision model is self.encoder.model.
        """
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

    def _forward_mambavision_all_levels(self, x):
        """
        Return all hierarchical feature maps from MambaVision.

        Typical mamba_vision_base shapes:
            outs[0]: [B, 256, 28, 28]
            outs[1]: [B, 512, 14, 14]
            outs[2]: [B, 1024, 7, 7]
            outs[3]: [B, 1024, 7, 7]
        """
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
        """
        Return late-stage feature maps:
            feat_a = levels[2]
            feat_b = levels[3] after final norm when available
        """
        model = self._get_mambavision_model()
        outs = self._forward_mambavision_all_levels(x)

        if len(outs) < 4:
            raise RuntimeError(f"Expected 4 MambaVision levels, got {len(outs)}")

        feat_a = outs[2]
        feat_b = outs[3]

        # Apply model.norm to the final stage, matching the standard MambaVision output.
        if hasattr(model, "norm") and model.norm is not None:
            try:
                feat_b = model.norm(feat_b)
            except Exception:
                pass

        feat_a = self._to_bchw(feat_a)
        feat_b = self._to_bchw(feat_b)

        return feat_a, feat_b

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
        """
        CroW-style non-parametric weighted pooling.

        feat: [B, C, H, W]
        Spatial and channel weights use abs(feat), while aggregation uses the original
        feature values to preserve sign information.
        """
        if feat.dim() != 4:
            raise RuntimeError(f"Expected feat [B,C,H,W], got {tuple(feat.shape)}")

        B, C, H, W = feat.shape
        feat_abs = torch.abs(feat)

        # Spatial weights emphasize locations with stronger responses.
        spatial = feat_abs.sum(dim=1, keepdim=True)  # [B,1,H,W]
        spatial = spatial / (spatial.mean(dim=(2, 3), keepdim=True) + eps)

        # Channel weights use an IDF-like count of locations above each channel mean.
        ch_mean = feat_abs.mean(dim=(2, 3), keepdim=True)  # [B,C,1,1]
        active = (feat_abs > ch_mean).float().sum(dim=(2, 3), keepdim=True)  # [B,C,1,1]

        num_pos = float(H * W)
        channel = torch.log((num_pos + eps) / (active + eps))  # [B,C,1,1]
        channel = channel / (channel.mean(dim=1, keepdim=True) + eps)

        weighted = feat * spatial * channel
        pooled = weighted.mean(dim=(2, 3))  # [B,C]
        return pooled

    def _power_l2_readout_from_maps(self, feat_a, feat_b):
        """
        GAP -> signed sqrt -> branch-wise L2 -> concat.
        Output: [B, 2C]
        """
        fa = self._gap(feat_a)
        fb = self._gap(feat_b)

        fa = self._branch_l2(self._signed_sqrt(fa))
        fb = self._branch_l2(self._signed_sqrt(fb))

        return torch.cat([fa, fb], dim=1)

    def _crow_readout_from_maps(self, feat_a, feat_b):
        """
        CroW pooling -> branch-wise L2 -> concat.
        Output: [B, 2C]
        """
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


class HisToGene_iBOTMamba(pl.LightningModule):
    """
    iBOT-MambaVision for gene expression prediction.

        Inputs:
        patch:  [B, 3, 224, 224]
        center: retained for API compatibility but currently unused
        exp:    [B, n_genes]

        Output:
        pred:   [B, n_genes]

        The gene prediction head maps aggregated features to expression values.
    """

    def __init__(
        self,
        n_genes=785,
        dim=640,
        learning_rate=1e-5,
        weight_decay=1e-4,
        checkpoint_path=None,
        arch="mamba_vision_tiny",
        freeze_encoder=True,
        feature_pool="HFAM",
    ):
        super().__init__()

        self.save_hyperparameters()

        self.n_genes = n_genes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.freeze_encoder = freeze_encoder
        self.arch = arch
        self.feature_pool = feature_pool

        self.ibot_mamba = IBOTMambaEncoder(
            checkpoint_path=checkpoint_path,
            arch=arch,
            strict=True,
            return_all_tokens=False,
            masked_im_modeling=False,
            feature_pool=feature_pool,
        )

        encoder_dim = self.ibot_mamba.output_dim

        if dim != encoder_dim:
            print(
                f"[HisToGene_iBOTMamba] Warning: given dim={dim}, "
                f"but encoder arch={arch}, feature_pool={feature_pool} outputs dim={encoder_dim}. "
                f"Use dim={encoder_dim} automatically."
            )
            dim = encoder_dim

        self.dim = dim

        if freeze_encoder:
            for p in self.ibot_mamba.parameters():
                p.requires_grad = False
            self.ibot_mamba.eval()

        self.gene_head = nn.Sequential(
            nn.Linear(dim, 1024),
            nn.Linear(1024, n_genes),
        )

    def forward(self, patches, centers=None):
        """
        patches:
            [B, 3, 224, 224]

        centers:
            Retained for compatibility with the original HisToGene dataloader; currently unused.
        """

        if self.freeze_encoder:
            self.ibot_mamba.eval()
            with torch.no_grad():
                h = self.ibot_mamba(patches)
        else:
            h = self.ibot_mamba(patches)

        pred = self.gene_head(h)
        return pred

    def training_step(self, batch, batch_idx):
        patch, center, exp = batch

        pred = self(patch, center)
        loss = F.mse_loss(pred.view_as(exp), exp)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        patch, center, exp = batch

        pred = self(patch, center)
        loss = F.mse_loss(pred.view_as(exp), exp)

        self.log(
            "valid_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def test_step(self, batch, batch_idx):
        patch, center, exp = batch

        pred = self(patch, center)
        loss = F.mse_loss(pred.view_as(exp), exp)

        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=50,
            gamma=0.9,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    @torch.no_grad()
    def extract_ibot_mamba_features(self, dataloader, device="cuda"):
        """
        Extract frozen iBOT-MambaVision features.

        Returns:
            features: [N, encoder_dim]
        """

        self.eval()
        self.to(device)

        all_features = []

        for batch in dataloader:
            patch = batch[0].to(device)

            h = self.ibot_mamba(patch)
            all_features.append(h.detach().cpu().numpy())

        features = np.concatenate(all_features, axis=0)
        return features

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--learning_rate", type=float, default=1e-5)
        parser.add_argument("--weight_decay", type=float, default=1e-4)
        parser.add_argument("--n_genes", type=int, default=785)
        parser.add_argument("--checkpoint_path", type=str, default=None)
        parser.add_argument("--arch", type=str, default="mamba_vision_tiny")
        parser.add_argument("--freeze_encoder", action="store_true")
        parser.add_argument("--finetune_encoder", action="store_true")
        parser.add_argument(
            "--feature_pool",
            type=str,
            default="mean",
            choices=IBOTMambaEncoder.VALID_POOLS,
        )

        return parser


def build_histogene_ibot_mamba(
    checkpoint_path,
    n_genes=785,
    learning_rate=1e-5,
    freeze_encoder=True,
    arch="mamba_vision_tiny",
    feature_pool="HFAM",
):
    """
    Build the gene expression prediction model.

    frozen encoder:
        freeze_encoder=True

    full fine-tuning:
        freeze_encoder=False
    """

    model = HisToGene_iBOTMamba(
        n_genes=n_genes,
        learning_rate=learning_rate,
        checkpoint_path=checkpoint_path,
        arch=arch,
        freeze_encoder=freeze_encoder,
        feature_pool=feature_pool,
    )

    return model
