Spatial gene expression prediction
==================================

Complete :doc:`installation` and :doc:`data`, then run the following two code blocks in order from the repository root. Set ``SGE_FOLD`` in the first block to choose a fold.

Training and testing
--------------------

.. literalinclude:: _includes/spatial_gene_expression_train_test.py
   :language: python

Visualization
-------------

.. literalinclude:: _includes/spatial_gene_expression_visualization.py
   :language: python

.. figure:: _static/figures/spatial_gene_expression.png
   :alt: H&E image, measured KLF6 expression, and SpaVista KLF6 prediction on P2 ST rep2
   :width: 95%

   cSCC fold-0 spatial expression example.
