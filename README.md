# GAN and VAE comparisons



## Generating samples from model

Example command for gan:

```bash
python gan.py sample --model_path runs/saved_models/best_gan.pt --n_samples 1000 --samples_dir results/samples/gan/images
```

Example command for vae:

```bash
python vae.py sample --model_path runs/saved_models/best_vae.pt --n_samples 1000 --samples_dir results/samples/vae/images
```

(Above commands used for generating samples used in FID calculation)

## Quantitative evaluations

Use [`score_fid.py`](score_fid.py) script for calculating FID between samples from svhn testset and model-generated samples saved in folder.

Thousand samples generated from the DCGAN and VAE model implemented here (trained on svhn trainset) are saved in [`results/samples/gan/images/`](results/samples/gan/images/) and [`results/samples/vae/images/`](results/samples/vae/images/) respectively.

## Qualitative evaluations

[`main.py`](main.py) is written for running qualitative evaluations of the trained gan and vae models.


## Reference

[DCGAN Paper](https://arxiv.org/pdf/1511.06434.pdf)