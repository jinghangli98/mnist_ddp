# minidp walkthrough (Part 0 answers)

Describes the **fixed** code (after `solution.patch`). Notes marked *(broken version)*
say what the original exercise code did instead.

## The one-paragraph version

`train.py` starts N identical processes. Each one builds the same model, takes a
different 1/N slice of the data, runs forward/backward on its own slice, then the
processes **average their gradients** so every optimizer step is identical on every rank.
That is all data parallelism is. The weights stay in sync not because anyone copies
weights around during training, but because everyone starts equal (one broadcast) and
applies the same averaged gradient every step.

## Mapping to what you already know

| your `train_ddp.py`                         | minidp                                             |
|---------------------------------------------|----------------------------------------------------|
| `mp.spawn(main, ...)`                       | `train.py`: `mp.spawn(worker, ...)`                |
| `ddp_setup()`                               | `dist_utils.setup()`                               |
| `DistributedSampler` + `DataLoader`         | `data.ShardedLoader` (both in one class)           |
| `train_sampler.set_epoch(epoch)`            | `train_loader.set_epoch(epoch)`                    |
| `DDP(model)` — broadcast at construction    | `DataParallel.__init__` — explicit `dist.broadcast`|
| `loss.backward()` — DDP hooks all-reduce grads invisibly | `loss.backward()` then an explicit `model.sync_grads()` |
| `dist.all_reduce(correct)` in `test()`      | `dist.all_reduce(stats)` in `evaluate()`           |
| `if rank == 0: torch.save(model.module...)` | `if is_main(): self.save_checkpoint(...)`          |

The only conceptual difference: PyTorch's DDP hides the gradient all-reduce inside
`backward()`. Here it is a visible function call.

## What each file does

### `train.py` — entry point
- `__main__`: parses the config; `world_size == 1` calls `worker(0, cfg)` directly,
  otherwise `mp.spawn` starts `world_size` processes, each running `worker(rank, cfg)`.
- `worker(rank, cfg)` — runs on **every** rank:
  1. `setup()` – join the process group
  2. `torch.manual_seed(cfg.seed + rank)` – each rank gets a *different* RNG stream,
     so `build_model` produces *different* initial weights on each rank…
  3. build loaders, build model, wrap in `DataParallel` – …which is why the wrapper
     must broadcast rank 0's weights. *(broken version: no broadcast → bug 3)*
  4. build SGD optimizer (each rank has its own, they stay identical because they see identical grads)
  5. `Trainer(...).fit()`, then `cleanup()`

### `minidp/config.py` — all hyperparameters
A dataclass; `from_cli()` auto-generates one `--flag` per field. Important ones:
`world_size`, `batch_size` (**per rank, per micro-batch**), `grad_accum_steps`,
`timeout_s` (how long a collective waits before raising — this is why hangs crash after 20 s).

### `minidp/dist_utils.py` — process-group plumbing
- `setup()`: sets `MASTER_ADDR/PORT` (where ranks find each other), picks `gloo` (CPU) or
  `nccl` (GPU), calls `init_process_group`. This call itself blocks until all ranks arrive.
- `cleanup()`, `get_device()`, `is_main()` (`rank == 0`), `log()` (print only on rank 0).

### `minidp/data.py` — dataset and sharding
- `TeacherDataset`: synthetic data. Seeded by a constant, **not** by rank, so every rank
  holds an identical copy of the full dataset in memory.
- `ShardedLoader._indices()` — the heart of sharding:
  1. build a permutation of all indices, seeded with `seed + epoch` → **the same
     permutation on every rank**, a new one each epoch
     *(broken version: `seed + epoch + rank` → different permutation per rank → bug 4)*
  2. `drop_last`: trim so it divides evenly by `world_size * batch_size`
  3. `order[rank::world_size]` → rank 0 takes positions 0,4,8…, rank 1 takes 1,5,9…
     Because the permutation is shared, these slices are disjoint and cover everything.
- `build_loaders()`: train loader shuffles + drops last; val loader does neither, so val
  shards are **uneven** (770 / 4 → 193, 193, 192, 192).

### `minidp/model.py` — a small residual MLP
Nothing distributed in here. No dropout/batchnorm, so runs are deterministic, which is
what lets `check.py` compare weights exactly.

### `minidp/parallel.py` — the DDP replacement
- `__init__`: `dist.broadcast(t.data, src=0)` for every parameter and buffer → all ranks
  start from rank 0's weights.
- `forward`: just calls the wrapped module. **No communication in forward.**
- `sync_grads()`: for each parameter, `all_reduce(SUM)` then divide by `world_size`
  = average gradient across ranks. *(broken version: no division → bug 1)*

### `minidp/trainer.py` — the loop
- `train_epoch()`: forward → `loss / accum` → `backward()` (local only). Every `accum`
  micro-batches: `sync_grads()` → `opt.step()` → `zero_grad()`, then all-reduce the loss
  purely for logging.
- `evaluate()`: each rank counts correct/total on its val shard into a 2-element tensor,
  then one `all_reduce` after the loop → exact global accuracy.
- `save_checkpoint()`: writes `ckpt.pt` (`model.module.state_dict()`, i.e. without the
  wrapper prefix) and `metrics.json`.
- `fit()`: per epoch: `set_epoch` → `train_epoch` → `evaluate` → rank 0 saves → `barrier`.

### `check.py` — the test
Runs `train.py` as subprocesses (1 rank × batch 32 as reference, then 4 × 8 and 2 × 16) and
compares final weights. Not part of the training code.

## Where communication happens

Every place ranks talk to each other, in execution order. **Everything not in this table
is purely local** — forward, backward, the optimizer step, data loading.

| # | where | call | how often | why |
|---|-------|------|-----------|-----|
| 1 | `dist_utils.setup` | `init_process_group` | once | rendezvous: ranks find each other |
| 2 | `parallel.DataParallel.__init__` | `broadcast(param, src=0)` per tensor | once | identical starting weights |
| 3 | `parallel.sync_grads` | `all_reduce(p.grad)` per parameter | once per **optimizer step** | average gradients — *the* DDP communication |
| 4 | `trainer.train_epoch` | `all_reduce(running)` | once per optimizer step | logging only; doesn't affect training |
| 5 | `trainer.evaluate` | `all_reduce(stats)` | once per epoch | global accuracy |
| 6 | `trainer.fit` | `barrier()` | once per epoch | others wait while rank 0 writes the checkpoint |
| 7 | `dist_utils.cleanup` | `destroy_process_group` | once | teardown |

The rule that explains every hang: **a collective only returns when every rank has called
it.** So each row above must be reached by all ranks, the same number of times, in the
same order. *(broken version: row 6 was inside `if is_main():` → only rank 0 called it → bug 2.)*
It's also why row 5 is *after* the eval loop: ranks have 4 vs 3 val batches, so a
collective inside the loop would be called an unequal number of times.

## What is the global batch size?

```
global batch = batch_size (per rank) × world_size × grad_accum_steps
default      = 8 × 4 × 1 = 32
```

It's the number of samples that contribute to **one optimizer step**. Each rank computes a
mean loss over its 8 samples; averaging 4 such gradients = the gradient of the mean over
all 32. That's why `check.py` compares against `--world_size 1 --batch_size 32`: same
global batch, so the math is identical. All of these are equivalent:

| world_size | batch_size | grad_accum_steps | global |
|-----------:|-----------:|-----------------:|-------:|
| 1 | 32 | 1 | 32 |
| 4 | 8  | 1 | 32 |
| 2 | 16 | 1 | 32 |
| 2 | 8  | 2 | 32 |
| 1 | 8  | 4 | 32 |

Steps per epoch = 4096 / 32 = 128 in every row (each rank iterates `128 × accum` micro-batches).

## Every rank vs. rank 0 only

**Every rank** (the default — assume this unless guarded):
- all of `worker()`: setup, seeding, building dataset / loaders / model / optimizer
- forward, backward, `sync_grads`, `opt.step()` — each rank keeps a full model copy and a
  full optimizer state
- the whole eval loop on its own val shard
- **every collective** in the table above
- `history.append(...)` — every rank keeps the list, only rank 0 ever writes it out

**Rank 0 only** (guarded by `is_main()`):
- `log(...)` printing — a bare `print()` would appear N times
- `save_checkpoint()` — writing `ckpt.pt` and `metrics.json`. Safe because weights are
  identical on all ranks, so rank 0's copy *is* the model. N ranks writing the same file
  would race.

**The pattern to remember:** side effects on the outside world (printing, files) → rank 0
only. Anything that computes or communicates → every rank. Never put a collective inside
an `if is_main():` block.

Note `val_acc` is computed on every rank (it needs everyone's all-reduce) even though only
rank 0 prints it — compute everywhere, report once.

## Try it: prints that make this concrete

```python
# train.py, in worker(), right after build_model / after DataParallel(...)
ck = sum(p.sum().item() for p in model.parameters())
print(f"[rank {rank}] param checksum {ck:.6f}", flush=True)
# before the wrapper: 4 different numbers. after: 4 identical numbers → that's the broadcast.

# trainer.py, in train_epoch around sync_grads (first step only)
g = next(self.model.parameters()).grad
print(f"[rank {dist.get_rank()}] grad norm before sync {g.norm():.4f}", flush=True)
self.model.sync_grads()
print(f"[rank {dist.get_rank()}] grad norm after  sync {g.norm():.4f}", flush=True)
# before: 4 different numbers (different data). after: identical.

# data.py, end of _indices()
print(f"[rank {self.rank}] epoch {self.epoch} first idx {order[self.rank::self.world_size][:6].tolist()}", flush=True)
# disjoint across ranks, changes each epoch.
```
