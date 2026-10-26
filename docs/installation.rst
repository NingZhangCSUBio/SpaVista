Installation
============

The validated environment is Linux x86_64 with Python 3.9.23, PyTorch 2.0.0+cu118, CUDA 11.8, and a compatible NVIDIA GPU and driver. The reported workflows were validated on an NVIDIA RTX A6000.

Clone the repository, then create and activate the environment:

.. code-block:: bash

   git clone --depth 1 https://github.com/NingZhangCSUBio/SpaVista.git
   cd SpaVista
   conda create -n SpaVista python=3.9.23 pip -y
   conda activate SpaVista
   python -m pip install -r requirements.txt
   python -m pip install --no-deps \
     'https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu118torch2.0cxx11abiFALSE-cp39-cp39-linux_x86_64.whl' \
     'https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu118torch2.0cxx11abiFALSE-cp39-cp39-linux_x86_64.whl' \
     'mambavision==1.2.0'
   python -m ipykernel install --user --name spavista --display-name "SpaVista"

Install MambaVision with ``--no-deps`` to retain the versions in ``requirements.txt``. These CUDA extension wheels are for Linux x86_64, Python 3.9, PyTorch 2.0, and CUDA 11.8. Typical installation time is 20–45 minutes on a Linux computer with a compatible NVIDIA GPU and broadband access.

Check the installation:

.. code-block:: bash

   python scripts/check_install.py
