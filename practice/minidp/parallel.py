import torch.distributed as dist
import torch.nn as nn


class DataParallel(nn.Module):
    """Bare-bones data parallelism on top of torch.distributed collectives.

    Unlike torch's DDP there are no autograd hooks: the trainer is expected to
    call `sync_grads()` between `backward()` and `optimizer.step()`.
    """

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def sync_grads(self):
        for p in self.module.parameters():
            if p.grad is None:
                continue
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
