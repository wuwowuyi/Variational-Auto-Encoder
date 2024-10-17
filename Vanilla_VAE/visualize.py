import glob

import numpy as np
import torch
import torchvision.datasets as dset
from torch import distributions as D

from matplotlib import pyplot as plt, gridspec
from torch.utils.data import DataLoader, sampler
from torchvision.transforms import v2 as T

from Vanilla_VAE.model import VAE
from train import save_samples, ckpt_path

image_dir = ckpt_path


def show_images(images):
    """
    Show images in grid.
    image.shape is (H, W, C) where C is the channel.
    """

    n_batch, batch_size = len(images), images[0].shape[0]

    fig = plt.figure(figsize=(n_batch, batch_size))
    gs = gridspec.GridSpec(n_batch, batch_size)
    gs.update(wspace=0.05, hspace=0.05)

    for i, batch in enumerate(images):
        for j in range(batch_size):
            idx = i * batch_size + j
            ax = plt.subplot(gs[idx])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(batch[j])
    plt.show()


def _prepare_file():
    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,)),
    ])
    mnist_train = dset.MNIST('./data', train=True, download=True, transform=transforms)
    loader_train = DataLoader(mnist_train, batch_size=10, sampler=sampler.SubsetRandomSampler(range(100)))
    for i, (x, _) in enumerate(loader_train):
        save_samples(torch.permute(x, (0, 2, 3, 1)), f'test_{i}')


def test_viz():
    _prepare_file()
    show_training_images("test")


def show_training_images(split: str = 'train'):
    images = []
    for f in sorted(glob.glob(f"{split}_*.npy", root_dir=image_dir)):
        images.append(np.load(image_dir / f))
    show_images(images)


def visualize_mnist(ckpt_file, *, zdim=20, sample="random"):
    f = ckpt_path / ckpt_file
    checkpoint = torch.load(f)
    model = VAE(28 * 28, 500, zdim)
    model.load_state_dict(checkpoint['model_dict'])

    if sample == 'manifold':
        assert zdim == 2
        xs = torch.linspace(0.025, 0.975, 20)
        ys = torch.linspace(0.025, 0.975, 20)
        x, y = torch.meshgrid(xs, ys, indexing='xy')
        cdfv = torch.stack((x.flatten(), y.flatten()), dim=1)
        norm = D.Normal(0.0, 1.0)
        z = norm.icdf(cdfv)
        batch = 20
    else:
        dist = D.Independent(D.Normal(torch.zeros(zdim), torch.ones(zdim)), 1)
        z = dist.sample((100,))
        batch = 10
    gen_x = model.sample(z)
    images = np.split(gen_x.numpy().reshape(-1, 28, 28, 1), batch)
    show_images(images)


if __name__ == "__main__":
    #test_viz()
    #show_training_images()
    ckpt_file = 'mnist-best-20'
    visualize_mnist(ckpt_file, zdim=20)

