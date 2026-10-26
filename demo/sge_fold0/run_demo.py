"""Run the cSCC-HVG fold-0 test on the bundled P2_ST_rep2 slide."""

import argparse
import hashlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DEMO = Path(__file__).resolve().parent
DATA = DEMO / "data" / "skin"
HEAD = DEMO / "checkpoints" / "fold_0_linear_head.pt"
TAG = "demo/cscc_hvg_fold0"
OUTPUT = ROOT / "results" / TAG
SLIDES = ["P2_ST_rep2"]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def one_match(pattern):
    matches = sorted((DATA / "GSE144240_RAW").glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one file matching {pattern}; found {len(matches)}")
    return matches[0]


def check_inputs():
    encoder = ROOT / "spavista" / "SpaVistav1.pth"
    if not encoder.is_file():
        raise FileNotFoundError(
            f"Missing {encoder}. Run: python scripts/download_hf_assets.py --checkpoint"
        )
    if not HEAD.is_file():
        raise FileNotFoundError(f"Missing {HEAD}. Run: python scripts/download_hf_assets.py --demo")
    gene_path = DATA / "skin_hvg_cut_1000.npy"
    if not gene_path.is_file():
        raise FileNotFoundError(gene_path)
    genes = np.load(gene_path, allow_pickle=True)
    if len(genes) != 171:
        raise ValueError(f"Expected 171 genes; found {len(genes)}")
    for slide in SLIDES:
        one_match(f"*{slide}.jpg")
        one_match(f"*spot*{slide}.tsv")
        parquet = DATA / "skin_ST-cnts_171" / f"{slide}_sub.parquet"
        if not parquet.is_file():
            raise FileNotFoundError(parquet)
    return encoder


def install_head():
    destination = OUTPUT / "linear_head_ckpt" / "fold_0_linear_head.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(destination) != sha256(HEAD):
            raise FileExistsError(f"Different checkpoint already exists: {destination}")
    else:
        shutil.copy2(HEAD, destination)


def minmax(values):
    values = np.asarray(values, dtype=np.float32)
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    return np.zeros_like(values) if high <= low else (values - low) / (high - low)


def save_figure(gene="KLF6", slice_index=1):
    slide = SLIDES[0]
    prefix = f"cscc_fold0_slice{slice_index}"
    predicted = np.load(OUTPUT / "npy_pred" / f"{prefix}_pred.npy")
    measured = np.load(OUTPUT / "npy_pred" / f"{prefix}_true.npy")
    genes = np.load(DATA / "skin_hvg_cut_1000.npy", allow_pickle=True).astype(str).tolist()
    gene_index = genes.index(gene)

    count = pd.read_parquet(DATA / "skin_ST-cnts_171" / f"{slide}_sub.parquet")
    position = pd.read_csv(one_match(f"*spot*{slide}.tsv"), sep="\t")
    position["id"] = (
        np.around(position["x"]).astype(int).astype(str)
        + "x"
        + np.around(position["y"]).astype(int).astype(str)
    )
    spots = count.join(position.set_index("id"), how="inner")
    xy = spots[["pixel_x", "pixel_y"]].to_numpy(dtype=float)
    if len(xy) != len(measured) or measured.shape != predicted.shape or measured.shape[1] != len(genes):
        raise ValueError("Spot or gene order differs from the saved prediction")

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(one_match(f"*{slide}.jpg")) as source_image:
        original_width, original_height = source_image.size
        source_image.draft("RGB", (2400, 2400))
        source_image.thumbnail((2400, 2400))
        image = source_image.convert("RGB")
    xy[:, 0] *= image.width / original_width
    xy[:, 1] *= image.height / original_height

    fig, axes = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)
    axes[0].imshow(image)
    axes[0].set_title(f"H&E - {slide}")
    for ax, values, title in zip(
        axes[1:],
        [measured[:, gene_index], predicted[:, gene_index]],
        ["Measured", "SpaVista"],
    ):
        points = ax.scatter(
            xy[:, 0], xy[:, 1], c=minmax(values), cmap="viridis",
            marker="o", s=12, vmin=0, vmax=1, linewidths=0,
        )
        ax.set_title(f"{title} {gene}")
        fig.colorbar(points, ax=ax, fraction=0.045)
    for ax in axes:
        ax.set_xlim(0, image.width)
        ax.set_ylim(image.height, 0)
        ax.axis("off")
    figure_dir = OUTPUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / f"{slide}_{gene}.png"
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-num", type=int, default=0, help="CUDA device index")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    encoder = check_inputs()
    install_head()
    command = [
        sys.executable, "-m", "tasks.spatial_gene_expression.run",
        "--config", str(ROOT / "configs" / "paths.yaml"),
        "--data_root", str(DATA),
        "--encoder_ckpt", str(encoder),
        "--dataset", "cscc",
        "--genes_type", "HVG",
        "--fold", "0",
        "--test_slice", "1",
        "--train_test", "False",
        "--tag", TAG,
        "--device_num", str(args.device_num),
        "--batchsize", str(args.batch_size),
        "--num_workers", "0",
    ]
    start = time.perf_counter()
    subprocess.run(command, cwd=ROOT, check=True)
    figure = save_figure()
    elapsed = time.perf_counter() - start

    metrics_path = OUTPUT / "median_performance_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    summary = metrics[["medpcc_gene", "med_spearman_gene"]].median()
    print(f"Gene PCC: {summary['medpcc_gene']:.4f}")
    print(f"Gene SCC: {summary['med_spearman_gene']:.4f}")
    print(f"Metrics: {metrics_path}")
    print(f"Figure: {figure}")
    print(f"Elapsed time: {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
