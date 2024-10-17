import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.datasets as dset
import wandb
import yaml
from torch import distributions as D
from torch import optim
from torch.nn import functional as F

from torch.utils.data import DataLoader, sampler
from torchvision.transforms import v2 as T

from model import VAE

USE_GPU = True
data_type = torch.float32
ckpt_path = Path(__file__).parent / 'checkpoint'
ckpt_path.mkdir(parents=True, exist_ok=True)

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = None


def get_dataloader(num_train, batch_size):
    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize((0.1307,), (1.0,)),
    ])

    mnist_train = dset.MNIST('./data', train=True, download=True, transform=transforms)
    loader_train = DataLoader(mnist_train, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(range(num_train)))
    # mnist_val = dset.MNIST('./data', train=True, download=True, transform=transforms)
    # loader_val = DataLoader(mnist_val, batch_size=16, sampler=sampler.SubsetRandomSampler(range(num_train, total_train)))
    # mnist_test = dset.MNIST('./data', train=False, download=True, transform=transforms)
    # loader_test = DataLoader(mnist_test, batch_size=batch_size)
    return loader_train


def sample(data_loader):
    x = next(iter(data_loader))
    x = x.to(device=device, dtype=data_type)
    out_dist = model(x)
    return out_dist.sample()


def save_samples(samples: torch.Tensor, fname: str) -> None:
    """For the convience of calling matplotlib imshow,
    a sample image's shape should be (H, W, C) where C is the channel.
    """
    samples = samples.detach().cpu().numpy()
    np.save(ckpt_path / f"{fname}.npy", samples)


def save_checkpoint(values: dict, ckpt_name: str) -> None:
    filename = f"{ckpt_name}_{str(int(time.time()))}.pkl"
    with open(ckpt_path / filename, "wb") as f:
        pickle.dump(values, f)


def train(config: dict, args: argparse.Namespace):

    seed = 1337 + args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    global model
    image_dim, hidden_dim, zdim = config['image_dim'], config['hidden_dim'], config['zdim']
    model = VAE(image_dim, hidden_dim, zdim, 10)
    model.to(device=device, dtype=data_type)

    num_train, batch_size = config['num_train'], config['batch_size']
    loader_train = get_dataloader(num_train, batch_size)

    optimizer = optim.Adam(model.parameters(), config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['decay_steps'], gamma=0.1)
    model.train()

    standard_normal = D.Normal(torch.zeros(zdim, device=device, dtype=data_type),
                               torch.ones(zdim, device=device, dtype=data_type))
    standard_normal = D.Independent(standard_normal, 1)


    num_epoch = config['total_steps'] // (num_train // batch_size)
    t = 0
    for epoch in range(num_epoch):
        for x, y in loader_train:
            x = x.reshape(batch_size, -1).to(device=device, dtype=data_type)  # shape=(batch_size, C * H * W)
            #y = y.to(device=device, dtype=torch.int64)
            gen_x, z_dist = model(x)

            a = D.kl_divergence(z_dist, standard_normal).mean()
            b = F.mse_loss(gen_x, x)
            loss = a + b

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # step optimizer
            scheduler.step(t)
            t += 1
            current_lr = scheduler.get_last_lr()[0]

            first_it = t == 1
            if first_it or t % args.log_interval == 0:
                if args.wandb_log:
                    wandb.log({
                        "step": t,
                        "loss": loss,
                        "lr": current_lr,
                    })
                else:
                    print(f"step {t}, lr {current_lr:.6f}, loss {loss:.4f}, kl-divergence {a:.4f}, cross-entropy {b:.4f}")

                if first_it or t % args.plot_interval == 0:
                    model.eval()
                    samples = gen_x.reshape(batch_size, *config['image_shape'])
                    idx = np.random.randint(100, size=(10,))
                    save_samples(samples[idx], f"train_{t}")
                    model.train()

    values = {
        'model_dict': model.state_dict()
    }
    save_checkpoint(values, config['model_name'])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--plot_interval", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb_log", action="store_true")
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    if args.wandb_log:  # wandb logging
        wandb_project = "vanilla-vae"
        wandb_run_name = f"{config['model_name']}-{str(int(time.time()))}"
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)

    print("training parameters:")
    for k, v in dict(**config, **vars(args)).items():
        print(f"{k}: {v}")

    train(config, args)


if __name__ == "__main__":
    main()
