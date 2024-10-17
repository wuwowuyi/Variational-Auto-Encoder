import glob

import numpy as np
import torch
import torchvision.datasets as dset
from matplotlib import pyplot as plt, gridspec
from torch.utils.data import DataLoader, sampler
from torchvision.transforms import v2 as T

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

    for i, batch in enumerate(reversed(images)):
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
    #_prepare_file()
    show_training_images("test")


def show_training_images(split: str = 'train'):
    images = []
    for f in sorted(glob.glob(f"{split}_*.npy", root_dir=image_dir)):
        images.append(np.load(image_dir / f))
    show_images(images)


if __name__ == "__main__":
    show_training_images()
