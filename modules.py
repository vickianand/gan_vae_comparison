import torch
from torch import nn
from itertools import chain
from functools import reduce


class GenericDiscriminator(nn.Module):
    """ a discriminator module for GAN which can also be used as convnet binary classifier
    Takes in input of shape (B, *image_dims) and returns a tensor of shape (B, 1)
    """

    def __init__(self, image_dims=(3, 32, 32), conv_out_channels=[64, 128, 256]):
        """ each Conv layer will halve the spatial dimensions
        """
        super().__init__()

        conv_layers_list = []
        in_channels = image_dims[0]
        for out_channels in conv_out_channels:
            conv_layers_list.append(
                nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)
            )
            conv_layers_list.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_layers_list)

        # calculate the output-spatial-dimensions
        flattened_conv_dim = (
            out_channels
            * (image_dims[1] // 2 ** len(conv_out_channels))
            * (image_dims[2] // 2 ** len(conv_out_channels))
        )
        # print(flattened_conv_dim)
        self.linear_layer = nn.Linear(flattened_conv_dim, 1)

    def forward(self, x):
        conv_out = self.conv_layers(x)
        return self.linear_layer(conv_out.view(conv_out.shape[0], -1))


class GenericGenerator(nn.Module):
    """ a generator module that can be used for a GAN or as a decoder in VAE
    Takes in input of shape (B, latent_dim) and returns a tensor of shape (B, *image_dims)
    """

    def __init__(
        self, latent_dim=100, image_dims=(3, 32, 32), conv_out_channels=[256, 128, 64]
    ):
        """ each Conv-transpose layer will double the spatial dimensions
        args:
            img_dims: shape of the image to be generated. It must have a
            3-dimensional shape and the spatial dimensions must be a 
            and power of 2

        """
        super().__init__()
        self.latent_dim = latent_dim
        self.first_3d_layer_shape = (
            conv_out_channels[0],
            (image_dims[1] // 2 ** len(conv_out_channels)),
            (image_dims[2] // 2 ** len(conv_out_channels)),
        )

        # calculate the output-spatial-dimensions
        flattened_conv_dim = reduce(lambda x, y: x * y, self.first_3d_layer_shape, 1)
        # print(flattened_conv_dim)

        self.linear_layer = nn.Linear(latent_dim, flattened_conv_dim)

        conv_trans_layers_list = []

        in_channels = conv_out_channels[0]
        for out_channels in chain(conv_out_channels[1:], image_dims[0:1]):
            conv_trans_layers_list.append(
                nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
            )
            conv_trans_layers_list.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        self.conv_trans_layers = nn.Sequential(*conv_trans_layers_list, nn.Tanh())
        # self.conv_trans_layers = nn.Sequential(*conv_trans_layers_list)

    def forward(self, z):
        assert z.shape[1] == self.latent_dim, "dimension of noise incorrect"
        linear_layer_out = self.linear_layer(z)
        return self.conv_trans_layers(
            linear_layer_out.view(z.shape[0], *self.first_3d_layer_shape)
        )


class ConvOnlyGenerator(nn.Module):
    """ Implement a DCGAN style conv-only generator for generating
    (n_channels, 32, 32) images from a given noise
    """

    def __init__(self, n_channels=3, latent_dim=100):
        """
        args:
            n_channels (int): number of channels in output image 
                (1 for greyscale, 3 for color)
            latent_dim: size for latent variable (noise)
        """
        super().__init__()
        self.latent_dim = latent_dim

        # first conv-transpose: (latent_dim, 1, 1) -> (1024, 4, 4)
        conv_trans_layers_list = [
            nn.ConvTranspose2d(latent_dim, 1024, 4, stride=1, padding=0)
        ]

        # sequence of three similar conv-transposes (each doubles the spatial dimension):
        # (1024, 4, 4) -> (512, 8, 8) -> (256, 16, 16) -> (n_channels, 32, 32)
        in_channels = 1024
        out_channels_list = [512, 256, n_channels]
        for out_channels in out_channels_list:
            conv_trans_layers_list.append(nn.BatchNorm2d(in_channels))
            conv_trans_layers_list.append(nn.ReLU(inplace=True))
            conv_trans_layers_list.append(
                nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
            )
            in_channels = out_channels

        self.net = nn.Sequential(*conv_trans_layers_list, nn.Tanh())

    def forward(self, z):
        z = z.view(z.shape[0], self.latent_dim, 1, 1)
        return self.net(z)


class ConvOnlyDiscriminator(nn.Module):
    """ Implement a DCGAN style conv-only discriminator to do binary
    classification of (n_channels, 32, 32) images
    """

    def __init__(self, n_channels=3, out_dim=1):
        """
        args:
            n_channels (int): number of channels in input image 
                (1 for greyscale, 3 for color)
        """
        super().__init__()

        # sequence of three similar conv layers (each halves the spatial dimension):
        # (n_channels, 32, 32) -> (256, 16, 16) -> (512, 8, 8)  -> (1024, 4, 4)
        conv_layers_list = []
        in_channels = n_channels
        out_channels_list = [256, 512, 1024]
        for out_channels in out_channels_list:
            conv_layers_list.append(
                nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)
            )
            conv_layers_list.append(nn.BatchNorm2d(out_channels))
            conv_layers_list.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = out_channels

        # last conv: (1024, 4, 4) -> (out_dim, 1, 1)
        self.net = nn.Sequential(
            *conv_layers_list,
            nn.Conv2d(out_channels_list[-1], out_dim, 4, stride=1, padding=0),
            # nn.Sigmoid() # not using sigmoid because using BCEWithLogitLoss()
        )

    def forward(self, z):
        return self.net(z).squeeze()


if __name__ == "__main__":
    pass
