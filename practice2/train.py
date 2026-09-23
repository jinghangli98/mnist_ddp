"""Entry point.

    python train.py                                   # single process, no DDP
    torchrun --standalone --nproc_per_node 4 train.py # 4 ranks on this machine
"""
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from tinyddp.config import Config
from tinyddp.data import build_loaders
from tinyddp.dist_utils import cleanup, is_distributed, is_main, setup
from tinyddp.model import build_model
from tinyddp.trainer import Trainer


def main(cfg):
    device = setup(cfg)
    torch.manual_seed(cfg.seed)

    train_loader, val_loader = build_loaders(cfg)
    model = build_model(cfg).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)

    state = {}
    if cfg.resume and is_main():
        # Only rank 0 touches the filesystem; DDP broadcasts the weights when it wraps
        # the model below, so the other ranks pick them up from there.
        state = torch.load(cfg.resume)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])

    if is_distributed():
        model = DDP(model)  # broadcasts rank 0's weights; all-reduces grads in backward()

    Trainer(cfg, model, optimizer, train_loader, val_loader, device, state).fit()
    cleanup()


if __name__ == "__main__":
    main(Config.from_cli())
