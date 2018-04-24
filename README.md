# FactorVAE

A TensorFlow implementation of the FactorVAE algorithm from the paper

Disentangling by Factorising (Kim & Mnih, 2018)
https://arxiv.org/pdf/1802.05983.pdf

You will need to download the file dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz from https://github.com/deepmind/dsprites-dataset and place it in the root of this repo.

To train the model, run

python factor_vae.py train

Checkpoints will be saved in ./checkpoints
The latest checkpoint can be loaded (for e.g. an interactive session) by running

python -i factor_vae.py load

