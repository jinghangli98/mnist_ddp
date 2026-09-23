import argparse
from dataclasses import dataclass, fields


@dataclass
class Config:
    # distributed — rank / world size / master addr come from torchrun's env vars (see dist_utils)
    device: str = "cpu"  # "cpu" -> gloo, "cuda" -> nccl, one GPU per rank
    timeout_s: int = 20  # how long a collective waits before raising
    seed: int = 0

    # data (synthetic teacher-student classification)
    n_train: int = 4096
    n_val: int = 770
    in_dim: int = 32
    n_classes: int = 10

    # model
    hidden: int = 128
    depth: int = 3
    aux_weight: float = 0.0  # weight of the auxiliary-head loss (0 = off)

    # optim
    batch_size: int = 8  # per-rank micro-batch size
    grad_accum_steps: int = 1
    lr: float = 0.0025
    momentum: float = 0.9
    epochs: int = 4
    target_acc: float = 0.76  # stop early once the global val accuracy reaches this

    # misc
    log_every: int = 32
    run_dir: str = "runs/default"
    resume: str = ""  # path to a ckpt.pt written by a previous run

    @classmethod
    def from_cli(cls, argv=None):
        p = argparse.ArgumentParser()
        for f in fields(cls):
            p.add_argument(f"--{f.name}", type=f.type, default=f.default)
        return cls(**vars(p.parse_args(argv)))
