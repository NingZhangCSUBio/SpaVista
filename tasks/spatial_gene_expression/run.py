# -*- coding: utf-8 -*-

import os
import random
import argparse
import warnings
import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from tasks.spatial_gene_expression.dataset import ViT_SKIN, ViT_HER2ST
from tasks.spatial_gene_expression.model import HisToGene_iBOTMamba
from spavista.config import load_path_config


# =========================
# Basic utils
# =========================
def seed_torch(seed=2024):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def check_and_create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created directory: {folder_path}")
    else:
        print(f"Directory already exists: {folder_path}")

def save_run_config(args, save_path):
    """
    Save all command-line arguments and basic runtime information.
    Files:
        results/<tag>/run_config.json
        results/<tag>/run_command.txt
    """
    os.makedirs(save_path, exist_ok=True)

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


def str2bool(v):
    """argparse bool parser. Supports True/False, yes/no, 1/0."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def make_auto_tag(args):
    """
    Generate result folder name when --tag is not provided.

    Format:
        spavista/<dataset>_<genes_type>_<feature_pool>_<max_epoch>epoch
    Example:
        ibot_mamba/cscc_heg_HFAM_20epoch
    """
    dataset = args.dataset.lower()
    genes_type = args.genes_type.lower()
    feature_pool = str(args.feature_pool)
    epoch = int(args.max_epoch)

    tag = f"spavista/{dataset}_{genes_type}_{feature_pool}_{epoch}epoch"
    return f"{tag}_fold{args.fold}" if args.fold is not None else tag


def infer_mambavision_dim(arch):
    dim_map = {
        "mamba_vision_tiny": 640,
        "mamba_vision_t2": 640,
        "mamba_vision_small": 768,
        "mamba_vision_base": 1024,
        "mamba_vision_large": 1568,
        "mamba_vision_l2": 1568,
    }

    if arch not in dim_map:
        raise ValueError(
            f"Unknown arch={arch}. Available archs: {list(dim_map.keys())}"
        )

    return dim_map[arch]


def infer_gene_num(dataset_name, genes_type):
    """Infer output gene number when --genes_num is not given."""
    dataset_name = dataset_name.lower()
    genes_type = genes_type.upper()

    gene_num_map = {
        ("cscc", "HEG"): 250,
        ("cscc", "HVG"): 171,
        ("her2st", "HEG"): 250,
        ("her2st", "HVG"): 785,
    }

    key = (dataset_name, genes_type)
    if key not in gene_num_map:
        raise ValueError(f"Unsupported dataset/genes_type combination: {key}")

    return gene_num_map[key]


def get_slice_num_list(dataset_name, genes_type):
    """
    Return number of test slices for each fold.

    cSCC: 4 patients/folds, 3 slices per patient.
    HER2ST-HEG: 8 folds.
    HER2ST-HVG: 7 folds, according to the published split.
    """
    dataset_name = dataset_name.lower()
    genes_type = genes_type.upper()

    if dataset_name == "cscc":
        return [3, 3, 3, 3]

    if dataset_name == "her2st":
        if genes_type == "HEG":
            return [6, 6, 6, 6, 3, 3, 3, 3]
        if genes_type == "HVG":
            return [5, 6, 6, 6, 3, 3, 3]

    raise ValueError(
        f"Unsupported dataset={dataset_name}, genes_type={genes_type}. "
        "dataset should be cscc/her2st, genes_type should be HEG/HVG."
    )


def get_dataset_class(dataset_name):
    dataset_name = dataset_name.lower()

    if dataset_name == "cscc":
        return ViT_SKIN
    if dataset_name == "her2st":
        return ViT_HER2ST

    raise ValueError(f"Unknown dataset={dataset_name}. Expected cscc or her2st.")


# =========================
# DataLoader for cSCC / HER2ST
# =========================
def get_train_loader(
    dataset_name,
    fold,
    batchsize,
    task_type,
    train_aug=True,
    normal=True,
    num_workers=4,
    data_root='data',
):
    dataset_cls = get_dataset_class(dataset_name)

    dataset = dataset_cls(
        train=True,
        fold=fold,
        patch_size=224,
        task_type=task_type,
        train_aug=train_aug,
        normal=normal,
        data_root=data_root,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=batchsize,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader


def get_test_loader(
    dataset_name,
    fold,
    test_slice,
    batchsize,
    task_type,
    train_aug=True,
    normal=True,
    num_workers=4,
    data_root='data',
):
    dataset_cls = get_dataset_class(dataset_name)

    dataset = dataset_cls(
        train=False,
        fold=fold,
        test_slice=test_slice,
        patch_size=224,
        task_type=task_type,
        train_aug=train_aug,
        normal=normal,
        data_root=data_root,
    )

    test_loader = DataLoader(
        dataset,
        batch_size=batchsize,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return test_loader


# =========================
# PyTorch Lightning trainer
# =========================
def get_pltrainer(trainepochs, device_num):
    """
    We disable Lightning checkpointing here.
    The encoder is frozen in the current setting, so we save only the trainable
    linear gene head with torch.save() after training. This avoids corrupted or
    empty .ckpt files from partial framework checkpoints.
    """
    trainer = pl.Trainer(
        logger=False,
        max_epochs=trainepochs,
        gpus=[device_num],
        enable_checkpointing=False,
    )

    return trainer


# =========================
# Linear-head checkpoint utils
# =========================
def get_linear_head_ckpt_path(tag, fold):
    return os.path.join(
        "results",
        tag,
        "linear_head_ckpt",
        f"fold_{fold}_linear_head.pt",
    )


def build_ibot_mamba_model(args, n_genes, encoder_dim):
    """Build model according to the current simplified logic: frozen encoder + Linear head."""
    model = HisToGene_iBOTMamba(
        n_genes=n_genes,
        dim=encoder_dim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        checkpoint_path=args.encoder_ckpt,
        arch=args.arch,
        freeze_encoder=True,
        feature_pool=args.feature_pool,
    )
    return model


def save_linear_head_checkpoint(model, args, fold):
    """
    Save only the Linear gene head weights and necessary metadata.

    Current logic always uses a frozen encoder, so the encoder can be rebuilt
    from args.encoder_ckpt + args.arch + args.feature_pool.
    """
    ckpt_path = get_linear_head_ckpt_path(args.tag, fold)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    payload = {
        "checkpoint_type": "linear_head_only",
        "fold": fold,
        "saved_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args).copy(),
        "model_config": {
            "n_genes": args.genes_num,
            "arch": args.arch,
            "feature_pool": args.feature_pool,
            "encoder_ckpt": args.encoder_ckpt,
            "freeze_encoder": True,
        },
        "gene_head_state_dict": {
            k: v.detach().cpu()
            for k, v in model.gene_head.state_dict().items()
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


def load_model_from_linear_head_checkpoint(args, fold, n_genes, encoder_dim, device="cpu"):
    """
    Rebuild the frozen encoder and load only the saved Linear head.
    """
    ckpt_path = get_linear_head_ckpt_path(args.tag, fold)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Cannot find linear head checkpoint: {ckpt_path}. "
            "Please train first or check --tag / --fold."
        )

    payload = torch.load(ckpt_path, map_location="cpu")
    if payload.get("checkpoint_type") != "linear_head_only":
        raise ValueError(f"Unsupported checkpoint type in {ckpt_path}: {payload.get('checkpoint_type')}")

    model = build_ibot_mamba_model(args, n_genes=n_genes, encoder_dim=encoder_dim)
    model.gene_head.load_state_dict(payload["gene_head_state_dict"], strict=True)
    model = model.to(device)
    model.eval()

    print(f"[INFO] Loaded linear head checkpoint from: {ckpt_path}")
    return model


# =========================
# Prediction
# =========================
@torch.no_grad()
def model_predict_ibot_mamba(
    model,
    test_loader,
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
):
    model.eval()
    model = model.to(device)

    preds = []
    gts = []

    for batch in tqdm(test_loader):
        # train=False returns:
        # patch, position, exp, center
        if len(batch) == 4:
            patch, position, exp, center = batch
        else:
            patch, position, exp = batch

        patch = patch.to(device)
        position = position.to(device)

        pred = model(patch, position)

        preds.append(pred.detach().cpu())
        gts.append(exp.detach().cpu())

    preds = torch.cat(preds, dim=0).squeeze().numpy()
    gt = torch.cat(gts, dim=0).squeeze().numpy()

    return preds, gt


# =========================
# Metrics
# =========================
def performance_med(pred, true, fold, test_slice, tag):
    """
    Median-based metrics.
    Returns:
        medpcc_cell,
        medpcc_gene,
        maxpcc_gene,
        med_heg50pcc_gene,
        med_hvg50pcc_gene,
        med_gene_mse,
        med_gene_mae,
        med_spearman_gene
    """

    # -------- cell-wise PCC --------
    corr_cell = np.zeros(pred.shape[0])
    for i in range(pred.shape[0]):
        corr_cell[i], _ = pearsonr(pred[i, :], true[i, :])
    corr_cell = corr_cell[~np.isnan(corr_cell)]

    medpcc_cell = np.median(corr_cell) if corr_cell.size else np.nan
    print(f"Median correlation across cells (PCC): {medpcc_cell:.4f}")

    # -------- gene-wise PCC --------
    corr_gene = np.zeros(pred.shape[1])
    p_values_gene = np.zeros(pred.shape[1])

    for j in range(pred.shape[1]):
        corr_gene[j], p_values_gene[j] = pearsonr(pred[:, j], true[:, j])

    valid_mask = ~np.isnan(corr_gene)
    corr_gene_valid = corr_gene[valid_mask]
    p_values_gene_valid = p_values_gene[valid_mask]

    medpcc_gene = np.median(corr_gene_valid) if corr_gene_valid.size else np.nan
    maxpcc_gene = np.max(corr_gene_valid) if corr_gene_valid.size else np.nan

    print(f"Median correlation across genes (PCC): {medpcc_gene:.4f}")
    print("Number of non-NaN genes: ", corr_gene_valid.shape[0])
    print(f"Max correlation (PCC): {maxpcc_gene:.4f}")

    # -------- save per-gene PCC --------
    gene_pcc_dir = os.path.join("results", tag, "gene_pcc")
    os.makedirs(gene_pcc_dir, exist_ok=True)

    df = pd.DataFrame(
        {
            "Gene_Index": np.arange(1, len(corr_gene_valid) + 1),
            "PCC": corr_gene_valid,
            "P_value": p_values_gene_valid,
        }
    )

    csv_path = os.path.join(
        gene_pcc_dir,
        f"genepcc_fold{fold}_slice{test_slice}.csv",
    )
    df.to_csv(csv_path, index=False)

    # -------- HEG/HVG top 50 median PCC --------
    ind_heg50 = np.argsort(np.sum(true, axis=0))[-50:]
    ind_hvg50 = np.argsort(np.var(true, axis=0))[-50:]

    heg50_vals = corr_gene[ind_heg50]
    heg50_vals = heg50_vals[~np.isnan(heg50_vals)]
    med_heg50pcc_gene = np.median(heg50_vals) if heg50_vals.size else np.nan

    hvg50_vals = corr_gene[ind_hvg50]
    hvg50_vals = hvg50_vals[~np.isnan(hvg50_vals)]
    med_hvg50pcc_gene = np.median(hvg50_vals) if hvg50_vals.size else np.nan

    print(f"Median PCC of top-50 highly expressed genes: {med_heg50pcc_gene:.4f}")
    print(f"Median PCC of top-50 highly variable genes: {med_hvg50pcc_gene:.4f}")

    # -------- per-gene MSE / MAE median --------
    gene_mse_list = []
    gene_mae_list = []

    for j in range(pred.shape[1]):
        gene_mse_list.append(mean_squared_error(true[:, j], pred[:, j]))
        gene_mae_list.append(mean_absolute_error(true[:, j], pred[:, j]))

    med_gene_mse = np.median(gene_mse_list) if len(gene_mse_list) else np.nan
    med_gene_mae = np.median(gene_mae_list) if len(gene_mae_list) else np.nan

    print(f"Median MSE across genes: {med_gene_mse:.4f}")
    print(f"Median MAE across genes: {med_gene_mae:.4f}")

    # -------- gene-wise Spearman median --------
    spearman_gene = np.zeros(pred.shape[1])

    for j in range(pred.shape[1]):
        spearman_gene[j], _ = spearmanr(pred[:, j], true[:, j])

    spearman_gene = spearman_gene[~np.isnan(spearman_gene)]
    med_spearman_gene = np.median(spearman_gene) if spearman_gene.size else np.nan

    print(f"Median Spearman correlation across genes: {med_spearman_gene:.4f}")

    return (
        medpcc_cell,
        medpcc_gene,
        maxpcc_gene,
        med_heg50pcc_gene,
        med_hvg50pcc_gene,
        med_gene_mse,
        med_gene_mae,
        med_spearman_gene,
    )


# =========================
# Train and test iBOT-Mamba
# =========================
def train_and_test_ibot_mamba(args, fold):
    n_genes = args.genes_num
    encoder_dim = infer_mambavision_dim(args.arch)

    if args.train_test:
        print("start training fold:", fold)
        print(f"Dataset: {args.dataset}")
        print(f"Gene type: {args.genes_type}")
        print(f"Number of genes: {n_genes}")
        print(f"Using arch: {args.arch}")
        print(f"Encoder dim: {encoder_dim}")
        print(f"Encoder checkpoint: {args.encoder_ckpt}")
        print(f"train_aug: {args.train_aug}")
        print(f"normal: {args.normal}")
        print(f"feature_pool: {args.feature_pool}")

        train_loader = get_train_loader(
            dataset_name=args.dataset,
            fold=fold,
            batchsize=args.batchsize,
            task_type=args.genes_type,
            train_aug=args.train_aug,
            normal=args.normal,
            num_workers=args.num_workers,
            data_root=args.data_root,
        )

        model = build_ibot_mamba_model(
            args=args,
            n_genes=n_genes,
            encoder_dim=encoder_dim,
        )

        trainer = get_pltrainer(
            trainepochs=args.max_epoch,
            device_num=args.device_num,
        )

        trainer.fit(model, train_loader)

        save_linear_head_checkpoint(model, args, fold)

    # ---------- test ----------
    print("start testing fold:", fold)

    slice_num_list = get_slice_num_list(args.dataset, args.genes_type)
    slices_for_fold = slice_num_list[fold]
    print(f"Dataset {args.dataset}, fold {fold} has {slices_for_fold} test slices.")

    device = torch.device(f"cuda:{args.device_num}") if torch.cuda.is_available() else torch.device("cpu")

    model = load_model_from_linear_head_checkpoint(
        args=args,
        fold=fold,
        n_genes=n_genes,
        encoder_dim=encoder_dim,
        device=device,
    )

    pred_save_dir = os.path.join("results", args.tag, "npy_pred")
    os.makedirs(pred_save_dir, exist_ok=True)

    all_fold_metrics = []

    for slice_idx in range(slices_for_fold):
        print(f"Testing slice {slice_idx} for fold {fold}.")

        test_loader = get_test_loader(
            dataset_name=args.dataset,
            fold=fold,
            test_slice=slice_idx,
            batchsize=args.batchsize,
            task_type=args.genes_type,
            train_aug=args.train_aug,
            normal=args.normal,
            num_workers=args.num_workers,
            data_root=args.data_root,
        )

        pred, truth = model_predict_ibot_mamba(
            model,
            test_loader,
            device=device,
        )

        pred_filename = f"{args.dataset}_fold{fold}_slice{slice_idx}_pred.npy"
        true_filename = f"{args.dataset}_fold{fold}_slice{slice_idx}_true.npy"
        np.save(os.path.join(pred_save_dir, pred_filename), np.asarray(pred))
        np.save(os.path.join(pred_save_dir, true_filename), np.asarray(truth))

        print("dataset:", args.dataset)
        print("fold:", fold)
        print("slice:", slice_idx)

        (
            medpcc_cell,
            medpcc_gene,
            maxpcc_gene,
            med_heg50pcc_gene,
            med_hvg50pcc_gene,
            med_gene_mse,
            med_gene_mae,
            med_spearman_gene,
        ) = performance_med(
            pred=pred,
            true=truth,
            fold=fold,
            test_slice=slice_idx,
            tag=args.tag,
        )

        all_fold_metrics.append(
            {
                "dataset": args.dataset,
                "genes_type": args.genes_type,
                "fold": fold,
                "slice_idx": slice_idx,
                "medpcc_cell": medpcc_cell,
                "medpcc_gene": medpcc_gene,
                "maxpcc_gene": maxpcc_gene,
                "med_heg50pcc_gene": med_heg50pcc_gene,
                "med_hvg50pcc_gene": med_hvg50pcc_gene,
                "med_gene_mse": med_gene_mse,
                "med_gene_mae": med_gene_mae,
                "med_spearman_gene": med_spearman_gene,
            }
        )

    return all_fold_metrics


# =========================
# Configuration
# =========================
def resolve_spatial_config(args, parser):
    """Resolve dataset, gene type, checkpoint, and dataset-specific root."""
    config = load_path_config(args.config)
    spatial = config.get("spatial_gene_expression", {}) or {}

    args.dataset = str(args.dataset or spatial.get("dataset", "cscc")).lower()
    args.genes_type = str(args.genes_type or spatial.get("genes_type", "HEG")).upper()
    args.encoder_ckpt = args.encoder_ckpt or config.get("checkpoint")

    if args.dataset not in {"cscc", "her2st"}:
        parser.error("dataset must be 'cscc' or 'her2st'.")
    if args.genes_type not in {"HEG", "HVG"}:
        parser.error("genes_type must be 'HEG' or 'HVG'.")

    if not args.data_root:
        roots = spatial.get("data_roots", {}) or {}
        root_key = "cscc" if args.dataset == "cscc" else "her2st"
        args.data_root = roots.get(root_key)
        if not args.data_root and args.dataset == "cscc":
            args.data_root = roots.get("skin")
        args.data_root = args.data_root or spatial.get("data_root")

    if not args.data_root:
        parser.error("Set spatial_gene_expression.data_roots in --config or pass --data_root.")
    if not args.encoder_ckpt:
        parser.error("Set checkpoint in --config or pass --encoder_ckpt.")
    return args


# =========================
# Main
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and test iBOT-Mamba on cSCC or HER2ST gene expression prediction."
    )

    parser.add_argument("--config", type=str, default="configs/paths.yaml", help="YAML file containing checkpoint and dataset-specific roots.")
    parser.add_argument("--data_root", type=str, default=None, help="Selected dataset directory; overrides the path in --config.")

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["cscc", "her2st"],
        help="Dataset name: cscc or her2st.",
    )

    parser.add_argument(
        "--device_num",
        type=int,
        default=0,
        help="GPU id.",
    )

    parser.add_argument(
        "--arch",
        type=str,
        default="mamba_vision_base",
        choices=[
            "mamba_vision_tiny",
            "mamba_vision_t2",
            "mamba_vision_small",
            "mamba_vision_base",
            "mamba_vision_large",
            "mamba_vision_l2",
        ],
        help="MambaVision encoder architecture.",
    )

    parser.add_argument(
        "--genes_num",
        type=int,
        default=None,
        help=(
            "Number of genes to predict. If omitted, it will be inferred: "
            "cscc HEG=250, cscc HVG=171, her2st HEG=250, her2st HVG=785."
        ),
    )

    parser.add_argument(
        "--genes_type",
        type=str,
        default=None,
        choices=["HVG", "HEG"],
        help="Gene type.",
    )

    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Run one zero-based fold; omit to run all folds.",
    )

    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help=(
            "Experiment name. Results will be saved to results/tag. "
            "If omitted, tag will be generated automatically as: "
            "spavista/<dataset>_<genes_type>_<feature_pool>_<max_epoch>epoch."
        ),
    )

    parser.add_argument(
        "--batchsize",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--max_epoch",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--train_test",
        type=str2bool,
        default=True,
        help="True: train and test. False: only test existing checkpoints.",
    )

    parser.add_argument(
        "--encoder_ckpt",
        type=str,
        default=None,
        help="SpaVista checkpoint path; overrides checkpoint in --config.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--train_aug",
        type=str2bool,
        default=True,
        help="Whether to use training augmentation in Dataset. Example: --train_aug True/False.",
    )

    parser.add_argument(
        "--normal",
        type=str2bool,
        default=True,
        help="Use ImageNet normalization (enabled by default).",
    )

    parser.add_argument(
        "--feature_pool",
        type=str,
        default="HFAM",
        choices=["HFAM"],
        help="Feature pooling mode.",
    )

    args = resolve_spatial_config(parser.parse_args(), parser)

    if args.genes_num is None:
        args.genes_num = infer_gene_num(args.dataset, args.genes_type)

    slice_num_list = get_slice_num_list(args.dataset, args.genes_type)
    fold_total = len(slice_num_list)
    if args.fold is not None and not 0 <= args.fold < fold_total:
        parser.error(f"--fold must be between 0 and {fold_total - 1} for {args.dataset} {args.genes_type}.")

    if args.tag is None or str(args.tag).strip() == "":
        args.tag = make_auto_tag(args)
        print(f"[INFO] Auto-generated tag: {args.tag}")
    else:
        print(f"[INFO] Using user-provided tag: {args.tag}")

    save_path = os.path.join("results", args.tag)
    check_and_create_folder(save_path)
    save_run_config(args, save_path)

    seed_torch(2024)

    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Gene type: {args.genes_type}")
    print(f"Gene number: {args.genes_num}")
    print(f"Fold total: {fold_total}")
    print(f"Slice nums per fold: {slice_num_list}")
    print(f"train_aug: {args.train_aug}")
    print(f"normal: {args.normal}")
    print(f"feature_pool: {args.feature_pool}")
    print("=" * 80)

    all_metrics = []

    selected_folds = range(fold_total) if args.fold is None else [args.fold]
    for fold in selected_folds:
        seed_torch(2024)

        print("=" * 80)
        print(f"Start fold {fold}")
        print("=" * 80)

        metrics = train_and_test_ibot_mamba(args, fold)

        for slice_metrics in metrics:
            all_metrics.append(slice_metrics)

    results = pd.DataFrame(all_metrics)
    results.to_csv(
        os.path.join("results", args.tag, "median_performance_metrics.csv"),
        index=False,
    )

    print("All folds finished.")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
