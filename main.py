"""
This script is written for running on cpu because here we only generate 
samples from a pre-trained models. For training the models use dcgan.py
and vae.py.
Also this script has full of hard-coded values and is written for 
for a very specific task (to generate outputs for report of a course assignment)
"""

import os
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import torch
from torchvision.utils import save_image

from data_utils import get_dataloader
from dcgan import DCGAN
from vae import VAE


def get_args():
    parser = ArgumentParser(
        description="program for doing some qualitative and quatitative evaluations of a gan and a vae model"
    )
    parser.add_argument(
        "--gan-model",
        default="runs/saved_models/best_dcgan.pt",
        help="Path to file saved by training dcgan model using dcgan.py script",
    )
    parser.add_argument(
        "--vae-model",
        default="runs/saved_models/best_vae.pt",
        help="Path to file saved by training vae model using vae.py script",
    )
    parser.add_argument(
        "--out-dir",
        default="results/",
        help="Directory where results of the comparison will be saved",
    )
    parser.add_argument("--seed", default=42, type=int, help="for random generrators")

    return parser.parse_args()


def main():
    """
    """
    args = get_args()
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    gan = DCGAN()
    gan.load_model(dict_path=args.gan_model)

    vae = VAE()
    vae.load_model(dict_path=args.vae_model)

    # ----------------------------------------------------------------------------------

    # # first save some random samples from both the models and also from original dataset
    # samples_dir = os.path.join(args.out_dir, "visual_samples/")
    # os.makedirs(samples_dir, exist_ok=True)

    # # # draw 3 8X8 grid of images from each of 3 sources
    # for i in range(1, 4):
    #     # original svhn dataset samples
    #     svhn_data_loader = get_dataloader("svhn_train", batch_size=64)
    #     orig_imgs, _ = next(iter(svhn_data_loader))
    #     save_image((orig_imgs * 0.5 + 0.5), samples_dir + f"orig_image_grid{i}.png")
    #     # gan samples
    #     gan_imgs = gan.sample(num_images=64)
    #     save_image(gan_imgs, samples_dir + f"gan_image_grid{i}.png")
    #     # gan samples
    #     vae_imgs = vae.sample(num_images=64)
    #     save_image(vae_imgs, samples_dir + f"vae_image_grid{i}.png")
    # # ----------------------------------------------------------------------------------

    # # next we want to see if the model has learned a disentangled representation in thelatent space
    disentg_dir = os.path.join(args.out_dir, "disentangled_repr/")
    os.makedirs(disentg_dir, exist_ok=True)
    imgs_per_row = 12
    eps = 15
    noise = torch.randn(imgs_per_row, 100)

    for tag, model in [("gan", gan), ("vae", vae)]:
        imgs_orig = model.sample(noise=noise)
        imgs_list = [imgs_orig, torch.zeros(imgs_per_row, 3, 32, 32)]
        interesting_dims = [14, 46, 51] if tag == "gan" else [12, 18, 70]
        # for i in tqdm(range(100)):
        for i in interesting_dims:
            noise_perturbed = noise.clone()
            noise_perturbed[:, i] += eps
            imgs_list.append(model.sample(noise=noise_perturbed))

        imgs_joined = torch.cat(imgs_list, dim=0)
        save_image(
            imgs_joined,
            disentg_dir + f"{tag}_disentang_3dims_seed{args.seed}_eps{eps}.png",
            nrow=imgs_per_row,
        )

    # ----------------------------------------------------------------------------------
    return

    # Compare between interpolations in the data space and in the latent space
    interpolations_dir = os.path.join(args.out_dir, "interpolations/")
    os.makedirs(interpolations_dir, exist_ok=True)
    z = torch.randn(2, 100)  # two noises which will be interpolated
    alpha = torch.linspace(0.0, 1.0, 11)  # .unsqueeze(1)  # unsqueeze for mat-mul
    z_interpolations = torch.ger(alpha, z[0]) + torch.ger((1 - alpha), z[1])
    alpha = alpha.view(-1, 1, 1, 1)  # so as to broadcast across 3-dimensional images
    for tag, model in [("gan", gan), ("vae", vae)]:
        x = model.sample(noise=z)
        imgs_x_interpolations = alpha * x[0] + (1 - alpha) * x[1]
        imgs_z_interpolations = model.sample(noise=z_interpolations)
        imgs_joined = torch.cat([imgs_x_interpolations, imgs_z_interpolations], dim=0)
        save_image(
            imgs_joined,
            interpolations_dir + f"{tag}_interpolations_s{args.seed}.png",
            nrow=11,
        )
    # ----------------------------------------------------------------------------------


if __name__ == "__main__":
    main()
