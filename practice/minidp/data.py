import torch

_DATA_SEED = 1234


class TeacherDataset:
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


class ShardedLoader:
    """Minimal DataLoader + DistributedSampler rolled into one.

    Every rank iterates over its own 1/world_size slice of the dataset.
    """

    def __init__(self, dataset, batch_size, rank, world_size, device,
                 shuffle=False, drop_last=False, seed=0):
        self.ds = dataset
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _indices(self):
        n = len(self.ds)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch + self.rank)
            order = torch.randperm(n, generator=g)
        else:
            order = torch.arange(n)
        if self.drop_last:
            order = order[: n - n % (self.world_size * self.batch_size)]
        return order[self.rank :: self.world_size]

    def __len__(self):
        n = len(self._indices())
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        idx = self._indices()
        for i in range(len(self)):
            b = idx[i * self.batch_size : (i + 1) * self.batch_size]
            yield self.ds.x[b].to(self.device), self.ds.y[b].to(self.device)


def build_loaders(cfg, rank, device):
    train_ds = TeacherDataset(cfg.n_train, cfg.in_dim, cfg.n_classes, "train")
    val_ds = TeacherDataset(cfg.n_val, cfg.in_dim, cfg.n_classes, "val")
    train_loader = ShardedLoader(train_ds, cfg.batch_size, rank, cfg.world_size, device,
                                 shuffle=True, drop_last=True, seed=cfg.seed)
    val_loader = ShardedLoader(val_ds, 64, rank, cfg.world_size, device)
    return train_loader, val_loader
