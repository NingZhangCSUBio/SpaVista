import torch
import scprep as scp
import os

HD_GENE = 'ERBB2'
hd_cfg = PATHS['visium_hd']
hd_root = Path(hd_cfg['base_dir'])
hd_slide = hd_cfg['test_slide']
hd_result = ROOT / 'results/visiumhd/SpaVista_10markers'
checkpoint = torch.load(hd_result / 'linear_head_ckpt/linear_head_last.pt', map_location='cpu')
hd_genes = list(checkpoint['model_config']['gene_names'])
gene_index = hd_genes.index(HD_GENE)
feature_dir = Path(hd_cfg['feature_dir'])
candidates = sorted(feature_dir.glob(f'features_{hd_slide}_test_*.npy'))
assert len(candidates) == 1, f'Expected one compatible cached feature file, found {candidates}'
features = np.load(candidates[0], mmap_mode='r')
def hd_expression_and_xy(slide):
    expression = pd.read_csv(hd_root / slide / hd_cfg['expression_file'])
    expression = expression.set_index(expression.columns[0])
    position = pd.read_parquet(hd_root / slide / 'subposition.parquet').set_index('barcode')
    meta = expression.join(position[['pxl_col_in_fullres', 'pxl_row_in_fullres']], how='left')
    assert not meta[['pxl_col_in_fullres', 'pxl_row_in_fullres']].isna().any().any()
    measured = scp.transform.log(scp.normalize.library_size_normalize(meta[hd_genes].to_numpy(dtype=np.float32)))[:, gene_index]
    xy = meta[['pxl_col_in_fullres', 'pxl_row_in_fullres']].to_numpy(dtype=np.float32)
    return measured, xy

measured_8, xy_8 = hd_expression_and_xy(hd_slide)
assert len(features) == len(measured_8)
prediction_file = FIGURES / f'{hd_slide}_{HD_GENE}_predicted_expression.npy'
inputs_time = max(candidates[0].stat().st_mtime, (hd_result / 'linear_head_ckpt/linear_head_last.pt').stat().st_mtime)
reuse_prediction = prediction_file.exists() and prediction_file.stat().st_mtime >= inputs_time
if reuse_prediction:
    prediction = np.load(prediction_file)
    reuse_prediction = prediction.shape == (len(features),)
if not reuse_prediction:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    state = checkpoint['gene_head_state_dict']
    first_w, first_b = state['0.weight'].float().to(device), state['0.bias'].float().to(device)
    second_w = state['1.weight'][gene_index].float().to(device)
    second_b = state['1.bias'][gene_index].float().to(device)
    temp_file = prediction_file.with_suffix('.tmp')
    output = np.lib.format.open_memmap(temp_file, mode='w+', dtype=np.float32, shape=(len(features),))
    with torch.no_grad():
        for start in range(0, len(features), 2048):
            stop = min(start + 2048, len(features))
            x = torch.from_numpy(np.asarray(features[start:stop], dtype=np.float32)).to(device)
            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                hidden = torch.nn.functional.linear(x, first_w, first_b)
                pred = hidden @ second_w + second_b
            output[start:stop] = pred.float().cpu().numpy()
    output.flush()
    del output
    os.replace(temp_file, prediction_file)
    prediction = np.load(prediction_file)

full_x = (float(xy_8[:, 0].min()), float(xy_8[:, 0].max()))
full_y = (float(xy_8[:, 1].min()), float(xy_8[:, 1].max()))
fig, axes = plt.subplots(1, 2, figsize=(6, 3.4), constrained_layout=True)
rows = [
    ('8 µm measured', measured_8, xy_8, 2.8),
    ('SpaVista prediction', prediction, xy_8, 2.8),
]
for ax, (label, values, xy, point_size) in zip(axes, rows):
    colors = minmax(values)
    points = ax.scatter(xy[:, 0], xy[:, 1], c=colors,
                        cmap='viridis', marker='o', s=point_size, vmin=0, vmax=1,
                        linewidths=0, rasterized=True)
    ax.set_xlim(*full_x); ax.set_ylim(full_y[1], full_y[0]); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(f'{label} {HD_GENE}')
fig.colorbar(points, ax=axes.tolist(), fraction=0.02, label='Min–max normalized expression')
saved_figure(f'visiumhd_{hd_slide}_{HD_GENE}.png')
