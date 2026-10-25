SGE_GENE = 'KLF6'
SGE_SLICE = 1
SGE_SLIDE = "P2_ST_rep2"
assert SGE_FOLD == 0, "P2_ST_rep2 visualization requires fold 0."
sge_cfg = PATHS['spatial_gene_expression']
assert sge_cfg['dataset'].lower() == 'cscc' and sge_cfg['genes_type'].upper() == 'HVG', 'This example expects cSCC-HVG results.'
skin = Path(sge_cfg['data_roots']['cscc'])
result = SGE_RESULT / 'npy_pred'
prefix = f'cscc_fold{SGE_FOLD}_slice{SGE_SLICE}'
true = np.load(result / f'{prefix}_true.npy')
pred = np.load(result / f'{prefix}_pred.npy')
genes = np.load(skin / 'skin_hvg_cut_1000.npy', allow_pickle=True).astype(str).tolist()
gene_index = genes.index(SGE_GENE)
cnt = pd.read_parquet(skin / 'skin_ST-cnts_171' / f'{SGE_SLIDE}_sub.parquet')
pos_file = sorted((skin / 'GSE144240_RAW').glob(f'*spot*{SGE_SLIDE}.tsv'))[0]
pos = pd.read_csv(pos_file, sep='\t')
pos['id'] = np.around(pos['x']).astype(int).astype(str) + 'x' + np.around(pos['y']).astype(int).astype(str)
spot = cnt.join(pos.set_index('id'), how='inner')
xy = spot[['pixel_x', 'pixel_y']].to_numpy(dtype=float)
assert len(xy) == len(true) == len(pred) and true.shape[1] == len(genes), 'Spot or gene order differs from the saved run.'
image_path = sorted((skin / 'GSE144240_RAW').glob(f'*{SGE_SLIDE}.jpg'))[0]
Image.MAX_IMAGE_PIXELS = None
image = Image.open(image_path)
original_width, original_height = image.size
image.draft('RGB', (2400, 2400))
image.thumbnail((2400, 2400))
image = image.convert('RGB')
xy[:, 0] *= image.width / original_width
xy[:, 1] *= image.height / original_height
fig, axes = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)
axes[0].imshow(image); axes[0].set_title(f'H&E — {SGE_SLIDE}')
for ax, values, title in zip(axes[1:], [true[:, gene_index], pred[:, gene_index]], ['Measured', 'SpaVista']):
    dots = ax.scatter(xy[:, 0], xy[:, 1], c=minmax(values), cmap='viridis', marker='o', s=12, vmin=0, vmax=1, linewidths=0)
    ax.set_title(f'{title} {SGE_GENE}')
    fig.colorbar(dots, ax=ax, fraction=0.045)
for ax in axes:
    ax.set_xlim(0, image.width); ax.set_ylim(image.height, 0); ax.axis('off')
saved_figure(f'sge_{SGE_SLIDE}_{SGE_GENE}.png')
