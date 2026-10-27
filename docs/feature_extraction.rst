Extract features from your own H&E images
=========================================

After :doc:`installation` and the encoder :doc:`download <data>`, run either command from the repository root. Inputs should be H&E image patches, not whole-slide images. Supported formats are ``.png``, ``.jpg``, ``.jpeg``, ``.tif``, ``.tiff``, ``.bmp``, and ``.webp``.

Single image
------------

.. code-block:: bash

   python scripts/extract_feature.py --image path/to/patch.tif --checkpoint spavista/SpaVistav1.pth --output patch_feature.npy --device cuda:0

This saves one feature vector in ``patch_feature.npy``.

One directory
-------------

.. code-block:: bash

   python scripts/extract_feature.py --image-dir path/to/he_patches --checkpoint spavista/SpaVistav1.pth --output my_spavista_features.npz --device cuda:0

This processes supported files directly inside the specified directory (not subdirectories). The ``.npz`` contains ``features`` and ``image_names``; each feature row corresponds to the name at the same index. TIFF images are opened as RGB patches; multi-page TIFFs use the first page.

Downstream models can be trained on these vectors for a defined prediction target. The five reproduction workflows use their own dataset formats and task-specific models.
