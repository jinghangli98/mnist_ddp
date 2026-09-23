import json
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .data import set_epoch
from .dist_utils import get_world_size, global_batch_size, is_distributed, is_main, log


class Trainer:
    def __init__(self, cfg, model, optimizer, train_loader, val_loader, device, state=None):
        self.cfg = cfg
        self.model = model
        self.opt = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        state = state or {}
        self.start_epoch = state.get("epoch", -1) + 1
        self.history = state.get("history", [])  # global mean loss, one entry per optimizer step
        self.global_step = state.get("global_step", 0)
        self.comm_calls = 0  # number of gradient all-reduces DDP has issued
        if is_distributed():
            self.model.register_comm_hook(self, _counting_allreduce)

    def loss_fn(self, batch):
        x, y = (t.to(self.device) for t in batch)
        logits, aux = self.model(x)
        # Per-sample losses normalised by the *global* batch, so the effective learning
        # rate does not depend on world size or accumulation.
        loss = F.cross_entropy(logits, y, reduction="sum") / global_batch_size(self.cfg)
        if self.cfg.aux_weight > 0:
            loss = loss + self.cfg.aux_weight * F.cross_entropy(aux, y)
        return loss

    def train_epoch(self, epoch):
        self.model.train()
        set_epoch(self.train_loader, epoch)
        accum = self.cfg.grad_accum_steps
        self.opt.zero_grad(set_to_none=True)
        running = torch.zeros((), device=self.device)
        for i, batch in enumerate(self.train_loader):
            loss = self.loss_fn(batch)
            loss.backward()  # grads accumulate across micro-batches
            running += loss.detach()
            if (i + 1) % accum != 0:
                continue
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)

            if is_distributed():
                dist.all_reduce(running)  # log the global mean loss, not this rank's
                running /= get_world_size()
            self.history.append(running.item())
            if self.global_step % self.cfg.log_every == 0:
                log(f"epoch {epoch} step {self.global_step:4d} loss {running.item():.4f}")
            running.zero_()
            self.global_step += 1

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        correct = torch.zeros((), device=self.device)
        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            logits, _ = self.model(x)
            correct += (logits.argmax(dim=1) == y).sum()
        if is_distributed():
            dist.reduce(correct, dst=0)  # rank 0 gets the total over all shards
        return correct.item() / len(self.val_loader.dataset)

    def save_checkpoint(self, epoch, val_acc):
        os.makedirs(self.cfg.run_dir, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.opt.state_dict(),
                "epoch": epoch,
                "global_step": self.global_step,
                "history": self.history,
            },
            os.path.join(self.cfg.run_dir, "ckpt.pt"),
        )
        with open(os.path.join(self.cfg.run_dir, "metrics.json"), "w") as f:
            json.dump({"loss": self.history, "val_acc": val_acc, "epoch": epoch,
                       "comm_calls": self.comm_calls}, f)

    def fit(self):
        log(f"{len(self.train_loader)} batches/epoch/rank | "
            f"global batch size {global_batch_size(self.cfg)} | "
            f"starting at epoch {self.start_epoch}")
        for epoch in range(self.start_epoch, self.cfg.epochs):
            self.train_epoch(epoch)
            val_acc = self.evaluate()
            log(f"=== epoch {epoch} done | val acc {val_acc:.4f} | "
                f"grad all-reduces so far {self.comm_calls}")
            if is_main():
                self.save_checkpoint(epoch, val_acc)
            if is_distributed():
                dist.barrier()  # nobody races ahead while rank 0 writes the checkpoint
            if val_acc >= self.cfg.target_acc:
                log(f"target accuracy {self.cfg.target_acc} reached, stopping")
                break


def _counting_allreduce(trainer, bucket):
    """DDP comm hook: the default all-reduce(mean), plus a counter."""
    trainer.comm_calls += 1
    fut = dist.all_reduce(bucket.buffer(), async_op=True).get_future()
    return fut.then(lambda f: f.value()[0] / get_world_size())
