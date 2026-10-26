Quick demo: cSCC-HVG fold 0
===========================

The small demo evaluates the held-out cSCC slide ``P2_ST_rep2`` with a trained fold-0 prediction head. It performs evaluation without retraining. Install SpaVista, then download the encoder checkpoint and demo assets from Hugging Face:

.. code-block:: bash

   python scripts/download_hf_assets.py --checkpoint --demo

From the repository root:

.. code-block:: bash

   python demo/sge_fold0/run_demo.py --device-num 0

``--device-num`` selects the CUDA-visible GPU index. The demo saves results in ``results/demo/cscc_hvg_fold0/``, including predictions, gene correlations, ``median_performance_metrics.csv``, and ``figures/P2_ST_rep2_KLF6.png``.

For ``P2_ST_rep2``, the gene PCC was 0.3209 and the gene SCC was 0.3160 on an NVIDIA RTX A6000. These are single-slide fold-0 demo metrics, not the four-fold aggregate.

To inspect the same workflow cell by cell, open `SpaVista_SGE_fold0_demo.ipynb <https://github.com/NingZhangCSUBio/SpaVista/blob/main/demo/sge_fold0/SpaVista_SGE_fold0_demo.ipynb>`_ with the ``SpaVista`` kernel.
