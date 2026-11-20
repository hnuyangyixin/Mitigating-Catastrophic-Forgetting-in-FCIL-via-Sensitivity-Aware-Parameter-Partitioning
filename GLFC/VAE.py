import numpy as np
import torch
from torch import nn


class VAE(nn.Module):
    """VAE模型用于FedCBC方法"""

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes):
        super().__init__()
        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size

        self.encoder = Encoder(encoder_layer_sizes, latent_size)
        self.decoder = Decoder(decoder_layer_sizes, latent_size)

    def forward(self, x):
        means, log_var = self.encoder(x)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z)
        return recon_x, means, log_var, z

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def generate(self, z):
        recon_x = self.decoder(z)
        return recon_x


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size):
        super().__init__()
        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.Tanh())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x):
        x = self.MLP(x)
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size):
        super().__init__()
        self.MLP = nn.Sequential()
        input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size] + layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i + 1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.Tanh())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z):
        x = self.MLP(z)
        return x