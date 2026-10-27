Extract features from your own H&E images
=========================================

After :doc:`installation` and the encoder :doc:`download <data>`, run either Python cell in a notebook started from the repository root. Each cell is independent. Use absolute paths if your notebook runs from another directory. Inputs should be H&E image patches, not whole-slide images. Supported formats are ``.png``, ``.jpg``, ``.jpeg``, ``.tif``, ``.tiff``, ``.bmp``, and ``.webp``.

Single image
------------

.. code-block:: python

   from pathlib import Path

   import numpy as np

   from spavista import SpaVistaFeatureExtractor

   image_path = Path("path/to/patch.tif")
   if not image_path.is_file():
       raise FileNotFoundError(image_path)

   extractor = SpaVistaFeatureExtractor("spavista/SpaVistav1.pth")
   feature = extractor.extract(image_path, device="cuda:0")
   np.save("patch_feature.npy", feature)
   print(feature.shape)

This saves one feature vector in ``patch_feature.npy``.

One directory
-------------

.. code-block:: python

   from pathlib import Path

   import numpy as np

   from spavista import SpaVistaFeatureExtractor

   image_dir = Path("path/to/he_patches")
   if not image_dir.is_dir():
       raise NotADirectoryError(image_dir)

   suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
   image_paths = sorted(
       (path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes),
       key=lambda path: path.name,
   )
   if not image_paths:
       raise FileNotFoundError(f"No supported image patches found in {image_dir}")

   extractor = SpaVistaFeatureExtractor("spavista/SpaVistav1.pth")
   features = np.stack([extractor.extract(path, device="cuda:0") for path in image_paths])
   np.savez(
       "my_spavista_features.npz",
       features=features,
       image_names=[path.name for path in image_paths],
   )
   print(features.shape)

This processes supported files directly inside the specified directory (not subdirectories). The ``.npz`` contains ``features`` and ``image_names``; each feature row corresponds to the name at the same index. TIFF images are opened as RGB patches; multi-page TIFFs use the first page.

Downstream models can be trained on these vectors for a defined prediction target. The five reproduction workflows use their own dataset formats and task-specific models.
