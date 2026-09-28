import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import os
import math
import contextlib


def ddp_setup():
    # torchrun sets MASTER_ADDR, MASTER_PORT, RANK, LOCAL_RANK, WORLD_SIZE for every process
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    init_process_group(backend='nccl')
    return local_rank, dist.get_rank()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def parse_args():
    p = argparse.ArgumentParser(description='MNIST model training')
    p.add_argument('--datadir', type=str, default='../data/', )
    p.add_argument('--epochs', type=int, default=20,)
    p.add_argument('--batch_size', type=int, default=64, )
    p.add_argument('--compile', action="store_true")
    p.add_argument('--lr', type=float, default=3e-4, )
    p.add_argument('--num_workers', type=int, default=8, )
    p.add_argument('--eval_freq', type=int, default=4,)
    p.add_argument('--grad_accum', type=int, default=1,
                   help='number of micro-batches to accumulate before an optimizer step')


    return p.parse_args()

def getloader(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(args.datadir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(args.datadir, train=False, transform=transform)

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=test_sampler)

    return train_loader, test_loader, train_sampler, test_sampler

def train(model, train_loader, optimizer, epoch, local_rank, rank, scheduler, grad_accum=1, log_interval=100):
    model.train()
    num_batches = len(train_loader)
    optimizer.zero_grad()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(local_rank), target.to(local_rank)
        # step on every grad_accum-th micro-batch, and on the last one so leftover grads are not dropped
        is_step = (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == num_batches

        # skip the all-reduce on micro-batches that don't step; DDP syncs on the stepping one
        sync_ctx = model.no_sync() if not is_step else contextlib.nullcontext()
        with sync_ctx:
            output = model(data)
            loss = torch.nn.functional.nll_loss(output, target)
            (loss / grad_accum).backward()

        if is_step:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if rank == 0 and batch_idx % log_interval == 0:
            print(f'Epoch: {epoch} [{batch_idx}/{num_batches}]'
                  f'loss {loss.item():.4f} lr {scheduler.get_last_lr()[0]:.2e}')

def test(model, test_loader, epoch, local_rank, rank):
    model.eval()

    correct = torch.zeros(1, device=local_rank)
    total = torch.zeros(1, device=local_rank)
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(local_rank), target.to(local_rank)
        with torch.no_grad():
            output = model(data)
            predictions = torch.argmax(torch.nn.Softmax(dim=1)(output), dim=1)
            correct += (predictions == target).sum()
            total += target.size(0)

    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    accuracy = (correct/total).item()

    if rank == 0:
        print(f'epoch: {epoch} | accuracy: {accuracy:.2f}')

def main(args):
    # local_rank: GPU index on this node (device placement)
    # rank: global rank across all nodes (logging / checkpointing)
    local_rank, rank = ddp_setup()

    train_loader, test_loader, train_sampler, test_sampler = getloader(args)
    model = Net().to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(0.05 * total_steps)
    warmup = LinearLR(
        optimizer,
        start_factor = 0.01,
        end_factor = 1,
        total_iters = warmup_steps
        )

    cosine = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min = 3e-6,
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps]
    )

    if rank == 0 :
        print(f'Training starts here....... (world size {dist.get_world_size()})')
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        train(model, train_loader, optimizer, epoch, local_rank, rank, scheduler, grad_accum=args.grad_accum)
        if epoch % args.eval_freq == 0:
            test(model, test_loader, epoch, local_rank, rank)

    if rank == 0:
        torch.save(model.module.state_dict(), 'model.pt')

    destroy_process_group()


if __name__ == '__main__':
    args = parse_args()
    main(args)
