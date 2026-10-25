import torch
from types import SimpleNamespace
from tasks.mif_generation import run as mifmod

MIF_TILE = '[8942,53607]_patch_6_7'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
args = SimpleNamespace(spavista_ckpt=PATHS['checkpoint'], spavista_arch='mamba_vision_base',
                       strict_load=False, num_feature_levels=4, image_size=224,
                       normal=True, amp='fp16' if device.type == 'cuda' else 'none')
dataset = mifmod.HEMITImageReconstructionDataset(
    csv_path=str(ROOT / PATHS['hemit']['labels_csv']), split='test',
    out_channels=['panCK', 'CD3'], data_root=str(PATHS['hemit']['root']), verbose=False)
matches = dataset.df.index[dataset.df['tile_id'].eq(MIF_TILE)].tolist()
assert len(matches) == 1, f'Expected one test tile named {MIF_TILE}, found {len(matches)}'
he_tensor, true_tensor, _ = dataset[matches[0]]
encoder, _, model_name = mifmod.build_encoder(args, 'spavista')
spatial_encoder = mifmod.MultiStageFeatureExtractor(encoder, model_name, num_feature_levels=4)
shapes = mifmod.infer_feature_shapes(spatial_encoder, args, device)
decoder = mifmod.MultiStageUNetDecoder(
    in_channels_list=[shape[0] for shape in shapes], out_channels=2,
    decoder_dim=128, image_size=224, num_post_blocks=2, norm='gn')
model = mifmod.FrozenMultiStageEncoderUNet(spatial_encoder, decoder)
mifmod.load_best_decoder(model, str(ROOT / 'results/hemit_mif_spavista_bs6_10epoch_mean_ssim/SpaVista'), device)
with torch.no_grad():
    with mifmod.autocast_context(args, device):
        pred_tensor = model(he_tensor.unsqueeze(0).to(device))[0].float().cpu().clamp(0, 1)
true_tensor = true_tensor.clamp(0, 1)
he_image = mifmod.tensor_to_numpy_img(mifmod.denormalize_he(he_tensor))
true_values, pred_values = true_tensor.numpy(), pred_tensor.numpy()
scales = np.maximum(np.percentile(true_values.reshape(2, -1), 99.8, axis=1), 1e-6)
def display_mif(values):
    enhanced = np.clip(values / scales[:, None, None], 0, 1) ** 0.65
    rgb = np.zeros((224, 224, 3), dtype=np.float32)
    rgb[..., :2] = enhanced.transpose(1, 2, 0)
    return rgb
true_display, pred_display = display_mif(true_values), display_mif(pred_values)
channel_pcc = [np.corrcoef(true_values[i].ravel(), pred_values[i].ravel())[0, 1] for i in range(2)]
fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.6), constrained_layout=True)
for ax, img, title in zip(axes, [he_image, true_display, pred_display],
                          ['H&E', f'True mIF — {MIF_TILE}', f'SpaVista (mean PCC={np.mean(channel_pcc):.3f})']):
    ax.imshow(img); ax.set_title(title, fontsize=8, pad=3); ax.axis('off')
saved_figure('mif_8942_53607_patch_6_7.png')
del model, decoder, spatial_encoder, encoder
if torch.cuda.is_available(): torch.cuda.empty_cache()
