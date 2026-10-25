from pathlib import Path
import glob
import json
import subprocess
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display, Image as NotebookImage

ROOT = Path.cwd()
if not (ROOT / 'configs' / 'paths.yaml').exists() and (ROOT.parent / 'configs' / 'paths.yaml').exists():
    ROOT = ROOT.parent
assert (ROOT / 'configs' / 'paths.yaml').exists(), 'Run this notebook from the SpaVista repository.'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from spavista.config import load_path_config
CONFIG = ROOT / 'configs' / 'paths.yaml'
PATHS = load_path_config(CONFIG)
RUN_TRAINING = True
FIGURES = ROOT / 'tutorial' / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)

def run_task(module, *options):
    command = [sys.executable, '-m', module, '--config', str(CONFIG), *options]
    print(' '.join(command))
    if RUN_TRAINING:
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"
        subprocess.run(command, cwd=ROOT, check=True, env=env)

def show_metrics(path, columns=None):
    path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f'{path} is missing. Run the task cell with RUN_TRAINING=True first.')
    frame = pd.read_csv(path)
    if columns is not None:
        frame = frame[columns]
    print(frame.to_string(index=False, float_format=lambda value: f'{value:.4f}'))

def saved_figure(name):
    path = FIGURES / name
    plt.savefig(path, dpi=220, bbox_inches='tight')
    print('Saved:', path)
    plt.close()
    display(NotebookImage(filename=str(path), width=800))

def minmax(values):
    values = np.asarray(values, dtype=np.float32)
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    return np.zeros_like(values) if high <= low else (values - low) / (high - low)
run_task('tasks.mif_generation.run')
show_metrics('results/hemit_mif_spavista_bs6_10epoch_mean_ssim/SpaVista/test_metrics_overall.csv',
             ['mean_pixel_pcc', 'mean_global_ssim', 'mean_pixel_mae', 'mean_psnr', 'best_epoch'])
