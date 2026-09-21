import torch
import torch.multiprocessing as mp

from minidp.config import Config
from minidp.data import build_loaders
from minidp.dist_utils import cleanup, get_device, setup
from minidp.model import build_model
from minidp.parallel import DataParallel
from minidp.trainer import Trainer


def worker(rank, cfg):
    setup(rank, cfg)
    device = get_device(rank, cfg)
    torch.manual_seed(cfg.seed + rank)  # decorrelate RNG streams across ranks

    train_loader, val_loader = build_loaders(cfg, rank, device)
    model = DataParallel(build_model(cfg).to(device))
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)

    Trainer(cfg, model, optimizer, train_loader, val_loader).fit()
    cleanup()


if __name__ == "__main__":
    cfg = Config.from_cli()
    if cfg.world_size == 1:
        worker(0, cfg)
    else:
        mp.spawn(worker, args=(cfg,), nprocs=cfg.world_size)
