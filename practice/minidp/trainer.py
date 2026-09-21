import json
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .dist_utils import is_main, log


class Trainer:
    def __init__(self, cfg, model, optimizer, train_loader, val_loader):
        self.cfg = cfg
        self.model = model
        self.opt = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.history = []  # one entry per optimizer step
        self.global_step = 0

    def train_epoch(self, epoch):
        self.model.train()
        for x, y in self.train_loader:
            loss = F.cross_entropy(self.model(x), y)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.model.sync_grads()
            self.opt.step()

            self.history.append(loss.item())
            if self.global_step % self.cfg.log_every == 0:
                log(f"epoch {epoch} step {self.global_step:4d} loss {loss.item():.4f}")
            self.global_step += 1

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        correct, total = 0, 0
        for x, y in self.val_loader:
            pred = self.model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
        return correct / total

    def save_checkpoint(self, epoch, val_acc):
        os.makedirs(self.cfg.run_dir, exist_ok=True)
        torch.save(
            {"model": self.model.module.state_dict(), "epoch": epoch},
            os.path.join(self.cfg.run_dir, "ckpt.pt"),
        )
        with open(os.path.join(self.cfg.run_dir, "metrics.json"), "w") as f:
            json.dump({"loss": self.history, "val_acc": val_acc}, f)

    def fit(self):
        log(f"{len(self.train_loader)} batches/epoch/rank | "
            f"global batch size {self.cfg.global_batch_size}")
        for epoch in range(self.cfg.epochs):
            self.train_epoch(epoch)
            val_acc = self.evaluate()
            if is_main():
                log(f"=== epoch {epoch} done | val acc {val_acc:.4f}")
                self.save_checkpoint(epoch, val_acc)
                dist.barrier()  # don't race ahead while the checkpoint is being written
