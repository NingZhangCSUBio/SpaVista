"""Extract an HFAM feature vector from one image."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spavista.features import SpaVistaFeatureExtractor


def main():
    parser = argparse.ArgumentParser(description="Extract one SpaVista HFAM image feature.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--checkpoint", required=True, help="SpaVista checkpoint path.")
    parser.add_argument("--output", default="spavista_feature.npy", help="Output .npy path.")
    parser.add_argument("--arch", default="mamba_vision_base", help="SpaVista architecture.")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--strict_load", action="store_true")
    args = parser.parse_args()

    extractor = SpaVistaFeatureExtractor(args.checkpoint, arch=args.arch, strict=args.strict_load)
    feature = extractor.extract(args.image, image_size=args.image_size, device=args.device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, feature)
    print(f"Saved SpaVista HFAM feature {feature.shape} to {output}")


if __name__ == "__main__":
    main()
