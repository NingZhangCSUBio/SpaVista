Extract features from your own H&E images
=========================================

After :doc:`installation` and the encoder :doc:`download <data>`, place RGB H&E patches in a directory. The following script is self-contained when run from the repository root; replace the input directory with your own path.

.. code-block:: python

   from pathlib import Path

   import numpy as np

   from spavista import SpaVistaFeatureExtractor

   image_dir = Path("path/to/your/he_patches")
   image_paths = sorted(
       path for path in image_dir.iterdir()
       if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
   )
   if not image_paths:
       raise FileNotFoundError(f"No H&E image patches found in {image_dir}")

   extractor = SpaVistaFeatureExtractor("spavista/SpaVistav1.pth")
   features = np.stack([
       extractor.extract(path, device="cuda:0") for path in image_paths
   ])
   np.savez(
       "my_spavista_features.npz",
       features=features,
       image_names=[path.name for path in image_paths],
   )
   print(features.shape)

The rows of ``features`` follow the order of ``image_names``. To extract one patch without creating a script, use:

.. code-block:: bash

   python scripts/extract_feature.py --image path/to/patch.png --checkpoint spavista/SpaVistav1.pth --output patch_feature.npy --device cuda:0

Downstream models can be trained on these vectors for a defined prediction target. The five reproduction workflows use their own dataset formats and task-specific models.
