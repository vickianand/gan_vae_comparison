import sys
import os

sys.path.append(os.getcwd())

from itertools import chain
from argparse import ArgumentParser
from tqdm import tqdm
import torch, torch.nn as nn
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

from modules import ConvOnlyGenerator, ConvOnlyDiscriminator
from data_utils import get_dataloader


class VAE:
    def __init__(self, n_channels=3, latent_dim=100, device="cpu"):
        """
        Args:
            n_channels (int): number of channels in image
            latent_dim (int): size of noise used as input to generator
            device (str): one of {"cpu", "cuda"}
        """
        self.enc = ConvOnlyDiscriminator(
            n_channels=n_channels, out_dim=2 * latent_dim
        ).to(device)
        self.dec = ConvOnlyGenerator(n_channels=n_channels, latent_dim=latent_dim).to(
            device
        )

        self.optim = torch.optim.Adam(
            chain(self.enc.parameters(), self.dec.parameters()), lr=3e-4
        )
        self.glob_it = 0  # gloabl training iteration count (across epochs and resuming)

        self.criterion = nn.BCEWithLogitsLoss()

        self.latent_dim = latent_dim
        self.device = device

    # Reconstruction (mean-squared error) + KL divergence losses
    def ELBO(self, recon_x, x, mu, logvar):
        """ return: -ELBO(= BCE + DKL) """
        MSE = ((x - recon_x) ** 2).sum()
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        DKL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE + DKL

    def train(self, data_loader, epochs=20, log_dir="runs/test/", log_freq=500):
        """ run training loop
        """
        tb_logger = SummaryWriter(log_dir=log_dir)
        self.log_dir = log_dir
        constant_noise = torch.randn(64, self.latent_dim, device=self.device)
        # to be used for tensborboard-logging only

        for ep in range(1, epochs + 1):

            print("\n", "=" * 35, f"training epoch {ep}", "=" * 35, "\n")

            for it, (imgs, _) in tqdm(enumerate(data_loader)):
                self.glob_it += 1

                imgs = imgs.to(self.device)
                enc_out = self.enc(imgs)
                mu, logvar = enc_out.chunk(chunks=2, dim=1)

                # reparamnetrizesation
                std = torch.exp(0.5 * logvar)
                e = torch.randn(std.shape, device=self.device)
                z = mu + e * std

                imgs_recon = self.dec(z)

                loss = self.ELBO(imgs_recon, imgs, mu, logvar)

                self.enc.zero_grad()
                self.dec.zero_grad()
                loss.backward()
                self.optim.step()

                tb_logger.add_scalar("train_loss", loss, self.glob_it)

                if self.glob_it % log_freq == 0:
                    # log some images to tensorboard
                    tb_logger.add_figure(
                        "samples", self.get_mXn_samples_grid(4, 4), self.glob_it
                    )
                    print(
                        f"epoch {ep}, iter {it} (total iter {self.glob_it}): train_loss = {loss}"
                    )

            # per epoch logging
            print(
                f"epoch {ep}, iter {it} (total iter {self.glob_it}): train_loss = {loss}"
            )
            tb_logger.add_images("epoch/sample1", self.sample(noise=constant_noise), ep)
            tb_logger.add_images("epoch/sample2", self.sample(num_images=64), ep)
            # save model at end of each epoch
            self.save_model(model_name=f"vae_ep{ep}.pt", idx=self.glob_it)

    def sample(self, num_images=4, noise=None):
        if noise is None:
            noise = torch.randn(num_images, self.latent_dim, device=self.device)

        self.dec.eval()
        with torch.no_grad():
            images = self.dec(noise).to("cpu")
        # change range from (-1, 1) to (0, 1)
        images = images * 0.5 + 0.5
        return images

    def get_mXn_samples_grid(self, m=3, n=3):

        images = self.sample(num_images=m * n)
        images = images.permute(0, 2, 3, 1)  # make channels last

        f, axarr = plt.subplots(m, n)
        plt.axis("off")

        for i in range(m):
            for j in range(n):
                axarr[i, j].imshow(images[i * m + j])
                axarr[i, j].imshow(images[i * m + j])
                axarr[i, j].imshow(images[i * m + j])
                axarr[i, j].imshow(images[i * m + j])

        return f

    def save_model(self, model_name="best_model.pt", idx=0):
        save_dir = os.path.join(self.log_dir, "saved_models")
        os.makedirs(save_dir, exist_ok=True)
        model_details = {
            "encoder_states": self.enc.state_dict(),
            "decoder_states": self.dec.state_dict(),
            "optim_states": self.optim.state_dict(),
            "idx": idx,
        }
        torch.save(model_details, os.path.join(save_dir, model_name))

    def load_model(self, dict_path="runs/test/saved_model/dcgan_ep1.pt"):
        model_details = torch.load(dict_path, map_location=self.device)
        self.enc.load_state_dict(model_details["encoder_states"])
        self.dec.load_state_dict(model_details["decoder_states"])
        self.optim.load_state_dict(model_details["optim_states"])
        self.glob_it = model_details["idx"]
        print(f"Successfuly loaded models and optims from {dict_path}")


def read_args():
    parser = ArgumentParser(description="program for training or sampling a vae")
    parser.add_argument(
        "train_or_sample",
        choices={"train", "sample"},
        help="Whether to train or sample",
    )
    parser.add_argument(
        "--resume_path",
        default="",
        help="if want to resumet training from a saved model and optim",
    )
    parser.add_argument(
        "--device", choices={"cpu", "cuda"}, default="cpu", help="device to run on"
    )
    parser.add_argument("--data", default="svhn_train", help="data to work with")
    parser.add_argument(
        "--data_dir",
        default="data/svhn/",
        help="directory where datafile can be found or saved",
    )
    parser.add_argument(
        "--log_dir", default="runs/test/", help="directory for keeping runs"
    )
    parser.add_argument("--batch_size", default=32, type=int, help="batch size")
    parser.add_argument("--model_path", default="runs/test/saved_models/dcgan_ep1.pt")
    parser.add_argument("--samples_dir", default="runs/test/samples/")
    parser.add_argument("--n_samples", default=100, type=int)
    return parser.parse_args()


def save_n_samples(model, samples_dir, n_samples=100):
    """ save n_samples number of images generated from model in sample_dir
    model must have a function model.sample(n_images) which returns a batch of n_images
    """
    os.makedirs(samples_dir, exist_ok=True)
    image_bacth_list = []
    batch_size = 100
    for i in range(0, n_samples, batch_size):
        image_bacth_list.append(model.sample(batch_size))
        print(f"sampled {i+batch_size} images.")

    images = torch.cat(image_bacth_list, dim=0)
    for i in range(n_samples):
        save_path = os.path.join(samples_dir, f"{i+1}.png")
        save_image(images[i], save_path, padding=0)


def main():
    args = read_args()
    vae = VAE(device=args.device)
    if args.train_or_sample == "train":
        data_loader = get_dataloader(
            args.data, data_dir=args.data_dir, batch_size=args.batch_size
        )
        if args.resume_path != "":
            vae.load_model(dict_path=args.resume_path)
        vae.train(data_loader, epochs=100, log_dir=args.log_dir)
    elif args.train_or_sample == "sample":
        vae.load_model(args.model_path)
        save_n_samples(vae, args.samples_dir, args.n_samples)


if __name__ == "__main__":
    main()
