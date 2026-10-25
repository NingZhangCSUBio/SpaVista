from tasks.tumor_detection.run import STNetDataClassifyLazy

TUMOR_SECTION = '23268_D1'
tumor_cfg = PATHS['tumor_detection']
test_folders = str(tumor_cfg['test_folders']).split(',')
dataset = STNetDataClassifyLazy(str(tumor_cfg['root_dir']), test_folders, train=False, train_aug=False, cache_images=False, verbose=False)
saved = pd.read_csv(ROOT / 'results/tumor_detection_linear_probe/SpaVista/test_predictions.csv')
assert len(dataset.samples) == len(saved)
assert np.array_equal(np.array([sample['label'] for sample in dataset.samples]), saved['y_true'].to_numpy()), 'Saved prediction order does not match dataset order.'
selected = [i for i, sample in enumerate(dataset.samples) if TUMOR_SECTION in Path(sample['tif_path']).stem]
assert selected, f'No test spots found on {TUMOR_SECTION}'
sample = dataset.samples[selected[0]]
xy = np.array([[dataset.samples[i]['x'], dataset.samples[i]['y']] for i in selected])
image = Image.open(sample['tif_path'])
original_width, original_height = image.size
image.thumbnail((2400, 2400))
image = image.convert('RGB')
xy = xy.astype(float)
xy[:, 0] *= image.width / original_width
xy[:, 1] *= image.height / original_height
fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), constrained_layout=True)
axes[0].imshow(image); axes[0].set_title('H&E')
for ax, values, title in [
    (axes[1], saved.iloc[selected]['y_true'].to_numpy(), 'Ground truth'),
    (axes[2], saved.iloc[selected]['y_pred_for_metrics'].to_numpy(), 'SpaVista'),
]:
    colors = np.where(values.astype(int) == 1, '#f3cb55', '#b9bbbd')
    ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=55, linewidths=0)
    ax.set_title(title)
for ax in axes:
    ax.set_xlim(0, image.width); ax.set_ylim(image.height, 0); ax.set_aspect('equal'); ax.axis('off')
saved_figure(f'tumor_{TUMOR_SECTION}.png')
