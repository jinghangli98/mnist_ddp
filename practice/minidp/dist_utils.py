import datetime
import os

import torch
import torch.distributed as dist


def setup(rank, cfg):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(cfg.master_port)
    if cfg.device == "cuda":
        torch.cuda.set_device(rank)
        backend = "nccl"
    else:
        torch.set_num_threads(1)
        backend = "gloo"
    dist.init_process_group(
        backend,
        rank=rank,
        world_size=cfg.world_size,
        timeout=datetime.timedelta(seconds=cfg.timeout_s),
    )


def cleanup():
    dist.destroy_process_group()


def get_device(rank, cfg):
    return torch.device("cuda", rank) if cfg.device == "cuda" else torch.device("cpu")


def is_main():
    return dist.get_rank() == 0


def log(msg):
    if is_main():
        print(msg, flush=True)
