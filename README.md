# SpaVista downstream reproduction

This repository provides the code required to evaluate the frozen **SpaVista** encoder on five downstream tasks.


## Installation

The validated environment uses Linux x86_64, Python 3.9.23, PyTorch 2.0.0+cu118, and CUDA 11.8. A compatible NVIDIA driver and GPU are required.

```bash
cd SpaVista
conda create -n SpaVista python=3.9.23 pip -y
conda activate SpaVista
python -m pip install -r requirements.txt
python -m pip install --no-deps \
  'https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu118torch2.0cxx11abiFALSE-cp39-cp39-linux_x86_64.whl' \
  'https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu118torch2.0cxx11abiFALSE-cp39-cp39-linux_x86_64.whl' \
  'mambavision==1.2.0'
```

Install MambaVision with `--no-deps` to preserve the pinned package versions. The CUDA extension wheels above target Python 3.9, PyTorch 2.0, and CUDA 11.8 on Linux x86_64.

This environment was validated on all five downstream tasks, including both Visium HD targets. `ipykernel` is included for notebook execution.

## Download checkpoint and data

The [SpaVista Hugging Face repository](https://huggingface.co/ZNfiona1997/SpaVista/tree/main) is the release location for the encoder checkpoint and dataset. The dataset must be published under a top-level `data/` directory with its structure unchanged. From the SpaVista code repository root, run:

```bash
python scripts/download_hf_assets.py --checkpoint
python scripts/download_hf_assets.py --data
```

The script places the checkpoint at `spavista/SpaVistav1.pth` and downloads `data/` into the repository root. It verifies the checkpoint against SHA-256 `8cb665b931341c21fe16ec7b7d455441869ba19deeedc6a901e4253ba71a83d9`. The data command requires the Hub repository to contain a top-level `data/` directory. For a different local layout, edit `configs/paths.yaml` as described below.

## Data layout

Place the pretrained checkpoint at `spavista/SpaVistav1.pth` and organize all datasets under a single `data/` directory in the SpaVista repository root:

```text
SpaVista/
├── spavista/
│   └── SpaVistav1.pth
└── data/
    ├── SGE/
    │   ├── her2st/
    │   │   ├── ST-cnts_250/
    │   │   ├── ST-cnts_785/
    │   │   ├── genes_her2st_250.npy
    │   │   ├── genes_her2st_hvg785.npy
    │   │   └── data/
    │   │       ├── ST-imgs/
    │   │       ├── ST-spotfiles/
    │   │       └── ST-pat/
    │   └── skin/
    │       ├── GSE144240_RAW/
    │       ├── skin_ST-cnts_250/
    │       ├── skin_ST-cnts_171/
    │       ├── genes_skin_250.npy
    │       └── skin_hvg_cut_1000.npy
    ├── tumor_detection/
    │   └── TRISTdata/
    │       ├── ST-spotfiles/grouped_files/<slide_id>/
    │       └── feature_cache/
    ├── protein/
    │   └── HEMIT/
    │       ├── hemit_abundance_labels.csv
    │       ├── train/{input,label}/
    │       ├── val/{input,label}/
    │       ├── test/{input,label}/
    │       └── feature_cache/
    └── HR/
        └── visiumHD_Breast_Cancer_1002/
            ├── all_res_he.tif
            ├── HVG200_MARKER10.npy
            ├── visium16/{exp_HVG200_MARKER10.csv,subposition.parquet}
            ├── visium8/{exp_HVG200_MARKER10.csv,subposition.parquet}
            └── cache_features/
```

The tree shows the files and directories used by the tasks; the dataset may contain additional files. `configs/paths.yaml` points to this layout. To use another location, edit the paths in that file. Relative paths are resolved from the repository root.

## Reproduction notebook

Open `notebooks/SpaVista_downstream_with_visualizations.ipynb` in a Jupyter-compatible editor to run the tasks step by step. For each task, a training-and-testing cell is followed by a visualization cell.

## Command-line reproduction

Alternatively, run the commands below from the repository root. All tasks read `configs/paths.yaml` by default. Results are saved under `results/` by default.

### 1. Spatial gene expression prediction

The default configuration runs cSCC-HVG. To run HER2ST or another gene panel, set `spatial_gene_expression.dataset` (`cscc` or `her2st`) and `genes_type` (`HEG` or `HVG`) in the selected YAML file.

```bash
python -m tasks.spatial_gene_expression.run
```

### 2. Tumor detection

```bash
python -m tasks.tumor_detection.run
```

### 3. Protein abundance prediction

```bash
python -m tasks.protein_abundance.run
```

### 4. H&E-to-mIF image generation

```bash
python -m tasks.mif_generation.run
```

### 5. Visium HD high-resolution inference

```bash
python -m tasks.high_resolution.run --target_task hvg200
python -m tasks.high_resolution.run --target_task marker10
```

## Feature caches

The Hugging Face `data/` release includes pre-extracted image features for tumor detection, protein abundance prediction, and Visium HD inference. The code uses a compatible cache when available. If you organize the datasets yourself and a compatible cache is not found, the code extracts and saves the features automatically; the first run will take longer.
