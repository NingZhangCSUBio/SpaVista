Checkpoint and data
===================

The released encoder checkpoint and companion data are in the `SpaVista Hugging Face repository <https://huggingface.co/ZNfiona1997/SpaVista/tree/main>`_. For the three-slide fold-0 demo, run this command from the code repository root:

.. code-block:: bash

   python scripts/download_hf_assets.py --checkpoint --demo

For all downstream tasks, download the encoder checkpoint and complete dataset:

.. code-block:: bash

   python scripts/download_hf_assets.py --checkpoint
   python scripts/download_hf_assets.py --data

The checkpoint is placed at ``spavista/SpaVistav1.pth``. The ``--data`` option downloads the complete dataset under ``data/``; the ``--demo`` option downloads only the demo assets under ``demo/sge_fold0/``. Keep the downloaded directory structure unchanged. The paths used by the downstream workflows are defined in ``configs/paths.yaml``; edit that file if your full dataset is elsewhere.

The task directories are:

.. code-block:: text

   data/
   ├── SGE/her2st/
   ├── SGE/skin/
   ├── tumor_detection/TRISTdata/
   ├── protein/HEMIT/
   └── HR/visiumHD_Breast_Cancer/

The tumor detection, protein abundance, and Visium HD workflows reuse compatible image-feature caches under their respective data directories. If a compatible cache is absent, the workflow extracts features and saves them there. The first run can therefore take longer.

For the detailed input tree, see the `repository README <https://github.com/NingZhangCSUBio/SpaVista#data-layout>`_.
