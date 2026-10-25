Quick demo: cSCC-HVG fold 0
===========================

The small demo includes three held-out cSCC slides and a trained fold-0 prediction head. It performs evaluation without retraining. Install SpaVista and download the encoder checkpoint first; the demo input slides and head are already in the GitHub repository.

From the repository root:

.. code-block:: bash

   python demo/sge_fold0/run_demo.py --device-num 0

``--device-num`` selects the CUDA-visible GPU index. The demo saves results in ``results/demo/cscc_hvg_fold0/``, including per-slide predictions, gene correlations, ``median_performance_metrics.csv``, and ``figures/P2_ST_rep2_KLF6.png``.

On an NVIDIA RTX A6000 with batch size 16 and GPU 1, the three-slide demo completed in 75.2 seconds. Its median gene PCC was 0.3209 and median gene SCC was 0.3160. These are fold-0 demo metrics, not the four-fold aggregate.

To inspect the same workflow cell by cell, open `SpaVista_SGE_fold0_demo.ipynb <https://github.com/NingZhangCSUBio/SpaVista/blob/main/demo/sge_fold0/SpaVista_SGE_fold0_demo.ipynb>`_ with the ``SpaVista`` kernel.
