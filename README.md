# SpaVista

This repository provides the SpaVista H&E image encoder, a feature-extraction interface, and reproduction workflows for five downstream tasks.

Tutorials and detailed instructions: [SpaVista Documentation](https://spavista.readthedocs.io/en/latest/).

## Installation

The validated environment uses Linux x86_64, Python 3.9.23, PyTorch 2.0.0+cu118, and CUDA 11.8. A compatible NVIDIA driver and GPU are required.

```bash
git clone --depth 1 https://github.com/NingZhangCSUBio/SpaVista.git
cd SpaVista
conda create -n SpaVista python=3.9.23 pip -y
conda activate SpaVista
python -m pip install -r requirements.txt
python -m pip install --no-deps \
  'https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu118torch2.0cxx11abiFALSE-cp39-cp39-linux_x86_64.whl' \
  'https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu118torch2.0cxx11abiFALSE-cp39-cp39-linux_x86_64.whl' \
  'mambavision==1.2.0'
python -m ipykernel install --user --name spavista --display-name "SpaVista"
```

Install MambaVision with `--no-deps` to preserve the pinned package versions. The CUDA extension wheels above target Python 3.9, PyTorch 2.0, and CUDA 11.8 on Linux x86_64.

This environment was validated on all five downstream tasks, including both Visium HD targets. `ipykernel` is included for notebook execution.

Typical installation time: **20–45 minutes** on a Linux computer with a compatible NVIDIA GPU and broadband connection.

## Download checkpoint and data

The [SpaVista Hugging Face repository](https://huggingface.co/ZNfiona1997/SpaVista/tree/main) contains the encoder checkpoint and data. From the SpaVista code repository root, choose the download you need.

For the fold-0 Quick demo, download the encoder checkpoint, three cSCC test slides, and their trained prediction head:

```bash
python scripts/download_hf_assets.py --checkpoint --demo
```

For all downstream tasks, download the encoder checkpoint and complete dataset:

```bash
python scripts/download_hf_assets.py --checkpoint
python scripts/download_hf_assets.py --data
```

The checkpoint is saved at `spavista/SpaVistav1.pth`. The `--data` option downloads the complete dataset to `data/`; the `--demo` option downloads only the Demo assets to `demo/sge_fold0/`. For a different full-dataset layout, edit `configs/paths.yaml` as described below.

## Quick demo

The [cSCC-HVG fold-0 demo notebook](demo/sge_fold0/SpaVista_SGE_fold0_demo.ipynb) evaluates three held-out test slides, `P2_ST_rep1`, `P2_ST_rep2`, and `P2_ST_rep3`, with a trained prediction head. It runs inference and evaluation only; no training is required. After installing the environment and downloading the Quick demo assets above, open the notebook with the `SpaVista` kernel and run its cells in order. It executes the same command from the repository root:

```bash
python demo/sge_fold0/run_demo.py --device-num 0
```

`--device-num` selects the CUDA-visible GPU index. Use `--device-num 1` for GPU 1; the run time below was recorded with GPU 1.

The demo writes these files under `results/demo/cscc_hvg_fold0/`:

```text
median_performance_metrics.csv
npy_pred/cscc_fold0_slice{0,1,2}_{pred,true}.npy
gene_pcc/genepcc_fold0_slice{0,1,2}.csv
figures/P2_ST_rep2_KLF6.png
```

The notebook displays the three slide metrics and the spatial expression figure for `P2_ST_rep2`. On the tested NVIDIA RTX A6000 (48 GB), the three-slide demo completed in **75.2 seconds** with batch size 16 on GPU 1. The median gene PCC across the slides was **0.3209** and the median gene SCC was **0.3160**. These are fold-0 demo results, not the four-fold aggregate; small numerical differences may occur on other systems.

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
        └── visiumHD_Breast_Cancer/
            ├── all_res_he.tif
            ├── HVG200_MARKER10.npy
            ├── visium16/{exp_HVG200_MARKER10.csv,subposition.parquet}
            ├── visium8/{exp_HVG200_MARKER10.csv,subposition.parquet}
            └── cache_features/
```

The tree shows the files and directories used by the tasks; the dataset may contain additional files. `configs/paths.yaml` points to this layout. To use another location, edit the paths in that file. Relative paths are resolved from the repository root.

## Use SpaVista on your own H&E images

After installing the environment and downloading the checkpoint, run either command from the repository root. Inputs should be H&E image patches, not whole-slide images. Supported formats are `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`, and `.webp`.

For one image:

```bash
python scripts/extract_feature.py --image path/to/patch.tif --checkpoint spavista/SpaVistav1.pth --output patch_feature.npy --device cuda:0
```

For all supported images directly inside one directory:

```bash
python scripts/extract_feature.py --image-dir path/to/he_patches --checkpoint spavista/SpaVistav1.pth --output my_spavista_features.npz --device cuda:0
```

The single-image command saves one feature vector as `.npy`. The directory command saves `features` and `image_names` in `.npz`; each feature row corresponds to the name at the same index. Subdirectories are not scanned. TIFF images are opened as RGB patches; multi-page TIFFs use the first page. The five reproduction workflows below use their own dataset formats and task heads.

## Reproduction notebook

For the short evaluation-only example, open `demo/sge_fold0/SpaVista_SGE_fold0_demo.ipynb`. For the complete downstream training, testing, and visualizations, open `tutorial/SpaVista_downstream_results_preview.ipynb` in a Jupyter-compatible editor. The notebook includes saved outputs and can be run cell by cell; each task's training-and-testing cell is followed by a visualization cell.

## Command-line reproduction

Alternatively, run the commands below from the repository root. All tasks read `configs/paths.yaml` by default. Results are saved under `results/` by default.

### 1. Spatial gene expression prediction

The default configuration runs cSCC-HVG. To run HER2ST or another gene panel, set `spatial_gene_expression.dataset` (`cscc` or `her2st`) and `genes_type` (`HEG` or `HVG`) in the selected YAML file.

```bash
python -m tasks.spatial_gene_expression.run
```

### 2. Visium HD high-resolution inference

```bash
python -m tasks.high_resolution.run --target_task hvg200
python -m tasks.high_resolution.run --target_task marker10
```

### 3. Protein abundance prediction

```bash
python -m tasks.protein_abundance.run
```

### 4. H&E-to-mIF image generation

```bash
python -m tasks.mif_generation.run
```

### 5. Tumor detection

```bash
python -m tasks.tumor_detection.run
```

## Feature caches

The Hugging Face `data/` release includes pre-extracted image features for tumor detection, protein abundance prediction, and Visium HD inference. The code uses a compatible cache when available. If you organize the datasets yourself and a compatible cache is not found, the code extracts and saves the features automatically; the first run will take longer.
