import argparse
from dataclasses import dataclass, fields


@dataclass
class Config:
    # distributed
    world_size: int = 4
    device: str = "cpu"  # "cpu" -> gloo, "cuda" -> nccl
    master_port: int = 29533
    timeout_s: int = 20
    seed: int = 0

    # data (synthetic teacher-student classification)
    n_train: int = 4096
    n_val: int = 770
    in_dim: int = 32
    n_classes: int = 10

    # model
    hidden: int = 128
    depth: int = 3

    # optim
    batch_size: int = 8  # per-rank micro-batch size
    grad_accum_steps: int = 1
    lr: float = 0.02
    momentum: float = 0.9
    epochs: int = 4

    # misc
    log_every: int = 32
    run_dir: str = "runs/default"

    @property
    def global_batch_size(self):
        return self.batch_size * self.world_size * self.grad_accum_steps

    @classmethod
    def from_cli(cls, argv=None):
        p = argparse.ArgumentParser()
        for f in fields(cls):
            p.add_argument(f"--{f.name}", type=f.type, default=f.default)
        return cls(**vars(p.parse_args(argv)))
