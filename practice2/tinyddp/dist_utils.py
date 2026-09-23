import datetime
import os

import torch
import torch.distributed as dist


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_distributed() else 0


def get_world_size():
    return dist.get_world_size() if is_distributed() else 1


def is_main():
    return get_rank() == 0


def log(msg):
    if is_main():
        print(msg, flush=True)


def setup(cfg):
    """Join the process group if we were launched by torchrun, else run single-process.

    torchrun sets RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR and MASTER_PORT for every
    process; init_process_group reads them from the environment.
    """
    if cfg.device == "cpu":
        torch.set_num_threads(1)  # tiny model: one thread per process is fastest + deterministic
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return torch.device(cfg.device)

    local_rank = int(os.environ["LOCAL_RANK"])
    if cfg.device == "cuda":
        torch.cuda.set_device(local_rank)
        backend, device = "nccl", torch.device("cuda", local_rank)
    else:
        backend, device = "gloo", torch.device("cpu")
    dist.init_process_group(backend, timeout=datetime.timedelta(seconds=cfg.timeout_s))
    return device


def cleanup():
    if is_distributed():
        dist.destroy_process_group()


def global_batch_size(cfg):
    return cfg.batch_size * get_world_size() * cfg.grad_accum_steps
