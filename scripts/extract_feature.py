"""Extract SpaVista features from one image or one directory of image patches."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spavista.features import SpaVistaFeatureExtractor

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def main():
    parser = argparse.ArgumentParser(description="Extract SpaVista HFAM image features.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="One H&E image patch.")
    source.add_argument("--image-dir", type=Path, help="Directory of H&E image patches (not recursive).")
    parser.add_argument("--checkpoint", required=True, help="SpaVista checkpoint path.")
    parser.add_argument("--output", type=Path, help="Output .npy for one image or .npz for a directory.")
    parser.add_argument("--arch", default="mamba_vision_base", help="SpaVista architecture.")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--strict_load", action="store_true")
    args = parser.parse_args()

    if args.image is not None:
        if not args.image.is_file():
            parser.error(f"Image file not found: {args.image}")
        image_paths = [args.image]
        output = args.output or Path("spavista_feature.npy")
        if output.suffix.lower() != ".npy":
            parser.error("--output must end in .npy when using --image")
    else:
        if not args.image_dir.is_dir():
            parser.error(f"Image directory not found: {args.image_dir}")
        image_paths = sorted(
            (path for path in args.image_dir.iterdir()
             if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
            key=lambda path: path.name,
        )
        if not image_paths:
            parser.error(f"No supported image patches found in {args.image_dir}")
        output = args.output or Path("spavista_features.npz")
        if output.suffix.lower() != ".npz":
            parser.error("--output must end in .npz when using --image-dir")

    extractor = SpaVistaFeatureExtractor(args.checkpoint, arch=args.arch, strict=args.strict_load)
    features = [extractor.extract(path, image_size=args.image_size, device=args.device)
                for path in image_paths]
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.image is not None:
        np.save(output, features[0])
        print(f"Saved SpaVista HFAM feature {features[0].shape} to {output}")
    else:
        feature_matrix = np.stack(features)
        np.savez(output, features=feature_matrix,
                 image_names=[path.name for path in image_paths])
        print(f"Saved {len(image_paths)} SpaVista HFAM features {feature_matrix.shape} to {output}")


if __name__ == "__main__":
    main()
