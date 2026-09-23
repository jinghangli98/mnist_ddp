import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from .dist_utils import get_rank, get_world_size

_DATA_SEED = 1234


class TeacherDataset(Dataset):
    """Synthetic classification set: labels come from a fixed random teacher MLP."""

    def __init__(self, n, in_dim, n_classes, split):
        g = torch.Generator().manual_seed(_DATA_SEED)
        w1 = torch.randn(in_dim, 8, generator=g) / in_dim ** 0.5
        w2 = torch.randn(8, n_classes, generator=g)
        g.manual_seed(_DATA_SEED + (1 if split == "train" else 2))
        self.x = torch.randn(n, in_dim, generator=g)
        self.y = (torch.tanh(self.x @ w1) @ w2).argmax(dim=1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def shard(dataset, shuffle, seed=0):
    """A sampler that gives this rank its 1/world_size slice of `dataset`.

    Explicit num_replicas / rank so it also works single-process (world size 1).
    """
    return DistributedSampler(dataset, num_replicas=get_world_size(), rank=get_rank(),
                              shuffle=shuffle, seed=seed)


def set_epoch(loader, epoch):
    """Re-seed a shuffling DistributedSampler so each epoch draws a new permutation."""
    if isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(epoch)


def build_loaders(cfg):
    train_ds = TeacherDataset(cfg.n_train, cfg.in_dim, cfg.n_classes, "train")
    val_ds = TeacherDataset(cfg.n_val, cfg.in_dim, cfg.n_classes, "val")
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=64, sampler=shard(val_ds, shuffle=False))
    return train_loader, val_loader
