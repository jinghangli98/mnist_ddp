import torch.nn as nn


class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, dim)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.act(self.fc(self.norm(x)))


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, depth, n_classes):
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden)
        self.blocks = nn.Sequential(*[Block(hidden) for _ in range(depth)])
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x):
        return self.head(self.blocks(self.embed(x)))


def build_model(cfg):
    return MLP(cfg.in_dim, cfg.hidden, cfg.depth, cfg.n_classes)
