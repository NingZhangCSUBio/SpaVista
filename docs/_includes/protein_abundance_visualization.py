PROTEIN_MARKER = 'panCK_mean'
PROTEIN_TEST_INDEX = 0
from tasks.protein_abundance.run import resolve_hemit_path
protein = pd.read_csv(ROOT / 'results/hemit_abundance_regression_spavista_bounded/SpaVista/test_predictions.csv')
labels = pd.read_csv(ROOT / PATHS['hemit']['labels_csv'])
test = labels.loc[labels['split'].eq('test')].dropna(subset=['panCK_mean', 'panCK_pos_area', 'CD3_mean', 'CD3_pos_area', 'he_path']).reset_index(drop=True)
data_root = Path(PATHS['hemit']['root'])
test['he_path'] = test['he_path'].map(lambda p: resolve_hemit_path(p, str(data_root), 'test', 'input'))
test['label_path'] = test['label_path'].map(lambda p: resolve_hemit_path(p, str(data_root), 'test', 'label'))
test = test.loc[test['he_path'].map(lambda p: Path(p).exists())].reset_index(drop=True)
assert len(test) == len(protein), 'Test sample order or filtering differs from the saved predictions.'
row = test.iloc[PROTEIN_TEST_INDEX]
prediction = protein.iloc[PROTEIN_TEST_INDEX]
assert np.isclose(float(row[PROTEIN_MARKER]), float(prediction['true_' + PROTEIN_MARKER]), atol=1e-5)
he = Image.open(row['he_path']).convert('RGB')
label = np.asarray(Image.open(row['label_path']))
panck = label[..., 0] if label.ndim == 3 else label
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9), constrained_layout=True)
axes[0].imshow(he); axes[0].set_title('H&E input')
axes[1].imshow(panck, cmap='magma'); axes[1].set_title('Measured panCK fluorescence')
for ax in axes: ax.axis('off')
fig.suptitle(f"panCK mean: measured {prediction['true_panCK_mean']:.4f}  |  SpaVista {prediction['pred_panCK_mean']:.4f}")
saved_figure(f'protein_test_{PROTEIN_TEST_INDEX}_panCK.png')
