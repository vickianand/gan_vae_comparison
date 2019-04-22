"""
This script is written for running on cpu because here we only generate 
samples from a pre-trained models. For training the models use dcgan.py
and vae.py.
"""

import os
from argparse import ArgumentParser
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

    dcgan = DCGAN()
    dcgan.load_model(dict_path=args.gan_model)

    vae = VAE()
    vae.load_model(dict_path=args.vae_model)

    # first save some random samples from both the models and also from original dataset
    samples_dir = os.path.join(args.out_dir, "visual_samples/")
    os.makedirs(samples_dir, exist_ok=True)

    # draw 3 8X8 grid of images from each of 3 sources
    for i in range(1, 4):
        # original svhn dataset samples
        svhn_data_loader = get_dataloader("svhn_train", batch_size=64)
        orig_imgs, _ = next(iter(svhn_data_loader))
        save_image((orig_imgs * 0.5 + 0.5), samples_dir + f"orig_image_grid{i}.png")
        # gan samples
        gan_imgs = dcgan.sample(num_images=64)
        save_image(gan_imgs, samples_dir + f"gan_image_grid{i}.png")
        # gan samples
        vae_imgs = vae.sample(num_images=64)
        save_image(vae_imgs, samples_dir + f"vae_image_grid{i}.png")


if __name__ == "__main__":
    main()
