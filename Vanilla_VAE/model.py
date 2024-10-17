import torch
from torch import nn
from torch import distributions as D
from torch.nn import functional as F


def _init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0, std=0.01)
        nn.init.zeros_(module.bias)

class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, zdim):
        super().__init__()

        self.input = nn.Linear(input_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.mean_head = nn.Linear(hidden_dim, zdim)
        self.std_head = nn.Linear(hidden_dim, zdim)

        self.apply(_init_weights)

    def forward(self, x):
        h = self.tanh(self.input(x))
        mean = self.mean_head(h)
        std = self.std_head(h)
        std = F.softplus(std)  # always positive
        return D.Independent(D.Normal(mean, std), 1)


class Decoder(nn.Module):

    def __init__(self, zdim, hidden_dim, image_dim):
        super().__init__()

        self.input = nn.Linear(zdim, hidden_dim)
        self.tanh = nn.Tanh()
        self.hidden = nn.Linear(hidden_dim, image_dim)

        self.apply(_init_weights)

    def forward(self, z):
        x = self.tanh(self.input(z))
        x = self.tanh(self.hidden(x))
        return x


class VAE(nn.Module):
    def __init__(self, image_dim, hidden_dim, zdim):
        super().__init__()
        self.encoder = Encoder(image_dim, hidden_dim, zdim)
        self.decoder = Decoder(zdim, hidden_dim, image_dim)

    def forward(self, images):
        """
        Input image should be flattened, i.e. images.shape=(batch_size, C * H * W)
        where image_dim = C * H * W
        """
        z_dist = self.encoder(images)
        z = z_dist.rsample()
        x = self.decoder(z)
        return x, z_dist

    @torch.no_grad()
    def sample(self, z):
        return self.decoder(z)

