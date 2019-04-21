import sys
import os

sys.path.append(os.getcwd())

from argparse import ArgumentParser
from tqdm import tqdm
import torch, torch.nn as nn
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

from modules import ConvOnlyGenerator, ConvOnlyDiscriminator
from data_utils import get_dataloader


class DCGAN:
    def __init__(self, n_channels=3, latent_dim=100, device="cpu"):
        """
        Args:
            n_channels (int): number of channels in image
            latent_dim (int): size of noise used as input to generator
            device (str): one of {"cpu", "cuda"}
        """
        self.d = ConvOnlyDiscriminator(n_channels=n_channels).to(device)
        self.g = ConvOnlyGenerator(n_channels=n_channels, latent_dim=latent_dim).to(
            device
        )

        self.d_optim = torch.optim.Adam(
            self.d.parameters(), lr=2e-4, betas=(0.5, 0.999)
        )
        self.g_optim = torch.optim.Adam(
            self.g.parameters(), lr=2e-4, betas=(0.5, 0.999)
        )

        self.criterion = nn.BCEWithLogitsLoss()

        self.latent_dim = latent_dim
        self.device = device

    def train(
        self, data_loader, epochs=20, d_train_freq=2, log_dir="runs/test/", log_freq=200
    ):
        """ run training loop
        """
        tb_logger = SummaryWriter(log_dir=log_dir)
        self.log_dir = log_dir
        batch_size = data_loader.batch_size
        labels_ones = torch.ones(batch_size, device=self.device)
        labels_zeros = torch.zeros(batch_size, device=self.device)
        # marged_targets = torch.cat([labels_ones, labels_zeros], dim=0)

        glob_it = 0
        for ep in range(1, epochs + 1):

            print("\n", "=" * 35, f"training epoch {ep}", "=" * 35, "\n")

            for it, (imgs, _) in tqdm(enumerate(data_loader)):
                glob_it += 1

                imgs = imgs.to(self.device)
                # train disriminator
                noise = torch.randn(batch_size, self.latent_dim, device=self.device)
                fake_imgs = self.g(noise)
                predictions_real = self.d(imgs)
                predictions_fake = self.d(fake_imgs)
                # merged_predictions = torch.cat([predictions_real, predictions_fake], 0)
                d_loss_real = self.criterion(predictions_real, labels_ones)
                d_loss_fake = self.criterion(predictions_fake, labels_zeros)
                d_loss = d_loss_real + d_loss_fake

                self.d.zero_grad()
                d_loss.backward()
                self.d_optim.step()

                tb_logger.add_scalar("d_loss/total", d_loss, glob_it)
                tb_logger.add_scalar("d_loss/real", d_loss_real, glob_it)
                tb_logger.add_scalar("d_loss/fake", d_loss_fake, glob_it)

                # train generator every d_train_freq iterations
                if glob_it % d_train_freq == 0:
                    noise = torch.randn(batch_size, self.latent_dim, device=self.device)
                    fake_imgs = self.g(noise)
                    predictions_fake = self.d(fake_imgs)
                    g_loss = self.criterion(predictions_fake, labels_ones)

                    self.d.zero_grad()
                    self.g.zero_grad()
                    g_loss.backward()
                    self.g_optim.step()

                    tb_logger.add_scalar("g_loss", g_loss, glob_it)

                if glob_it % log_freq == 0:
                    # log some images to tensorboard
                    tb_logger.add_figure("samples", self.get_mXn_samples(4, 4), glob_it)
                    # tb_logger.add_images("samples", self.sample(9), glob_it)

                    print(
                        f"epoch {ep}, iter {it}: d_loss = {d_loss}, g_loss = {g_loss}"
                    )

            # per epoch logging
            print(f"epoch {ep}, iter {it}: d_loss = {d_loss}, g_loss = {g_loss}")
            tb_logger.add_images("epoch_samples", self.sample(64), ep)
            # save model at end of each epoch
            self.save_model(model_name=f"dcgan_ep{ep}.pt", idx=glob_it)

    def sample(self, num_images=4):
        with torch.no_grad():
            noise = torch.randn(num_images, self.latent_dim, device=self.device)
            images = self.g(noise).to("cpu")
        # change range from (-1, 1) to (0, 1)
        images = images * 0.5 + 0.5
        return images

    def get_mXn_samples(self, m=3, n=3):

        images = self.sample(num_images=m * n)
        images = images.permute(0, 2, 3, 1)  # make channels last

        f, axarr = plt.subplots(m, n)
        for i in range(m):
            for j in range(n):
                axarr[i, j].imshow(images[i * m + j])
                axarr[i, j].imshow(images[i * m + j])
                axarr[i, j].imshow(images[i * m + j])
                axarr[i, j].imshow(images[i * m + j])

        plt.axis("off")
        return f

    def save_model(self, model_name="best_model.pt", idx=0):
        save_dir = os.path.join(self.log_dir, "saved_models")
        os.makedirs(save_dir, exist_ok=True)
        model_details = {
            "discriminator_states": self.d.state_dict(),
            "generator_states": self.g.state_dict(),
            "d_optim_states": self.d_optim.state_dict(),
            "g_optim_states": self.g_optim.state_dict(),
            "idx": idx,
        }
        torch.save(model_details, os.path.join(save_dir, model_name))

    def load_model(self, dict_path="runs/test/saved_model/dcgan_ep1.pt"):
        model_details = torch.load(dict_path, map_location=self.device)
        self.d.load_state_dict(model_details["discriminator_states"])
        self.g.load_state_dict(model_details["generator_states"])
        self.d_optim.load_state_dict(model_details["d_optim_states"])
        self.d_optim.load_state_dict(model_details["g_optim_states"])


def read_args():
    parser = ArgumentParser(description="program for training or sampling a dc-gan")
    parser.add_argument(
        "train_or_sample",
        choices={"train", "sample"},
        help="Whether to train or sample",
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
    gan = DCGAN(device=args.device)
    if args.train_or_sample == "train":
        data_loader = get_dataloader(
            args.data, data_dir=args.data_dir, batch_size=args.batch_size
        )
        gan.train(data_loader, epochs=100, log_dir=args.log_dir)
    elif args.train_or_sample == "sample":
        gan.load_model(args.model_path)
        save_n_samples(gan, args.samples_dir, args.n_samples)


if __name__ == "__main__":
    main()
