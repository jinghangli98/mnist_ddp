# Mock interview 2: `tinyddp`

Second practice set. The first one (`../practice`) implemented data parallelism by hand;
this one uses the real stack, which is what you'll most likely be handed in the interview:
`DistributedDataParallel`, `DataLoader` + `DistributedSampler`, `torchrun` env-var
rendezvous, checkpoint / resume. CPU + `gloo`, a full run takes ~6 s.

**Rules (same as the real thing):** 45-minute timer, no AI tools, PyTorch docs allowed,
talk out loud, add lots of `print()`s. Don't open `solutions/` until you're done.

```bash
cd practice2
python train.py --batch_size 32                        # single process, no DDP: works
torchrun --standalone --nproc_per_node 4 train.py      # 4 ranks: ...does not go well
python check.py                                        # objective pass/fail for part 1
```

`python train.py` runs single-process (no process group, no DDP wrapper). `torchrun`
sets `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT` for each process
and `setup()` joins the group. This "if distributed" split is where most real-world
DDP bugs hide, because the single-process path never exercises them.

## Part 0 — read (5 min)

Before touching anything, be able to say out loud:

- What does `torchrun` do that `python train.py` doesn't, and what does `setup()` read?
- Every place ranks communicate, *including the ones hidden inside `DDP`* (what happens at
  construction, in `forward`, in `backward`).
- The global batch size, and how many optimizer steps per epoch you expect on 4 ranks.
- What runs on every rank vs. rank 0 only. What's in the checkpoint.
- What `_counting_allreduce` (a DDP *comm hook*) is for. You'll use it in part 2.

## Part 1 — debug (20 min)

A colleague says "single-GPU works, the multi-GPU launch is a mess". The invariant:

> With the same global batch size, training on N ranks yields the same weights as
> training on 1 rank (up to float noise), *and the checkpoints are interchangeable*.

`python check.py` must print `PASS` for both setups. There are **five** independent bugs.
One crashes, one hangs-then-crashes, two are silent, one you only notice when you load
the checkpoint. For each one, *show* it with a print before fixing it.

Things to print (from every rank, rank in the message, `flush=True`):
`len(train_loader)`; the labels of the first batch (`y[:8].tolist()`); the gradient norm
after `backward()` at step 0 (compare 1 rank vs 4); `val_acc` from every rank;
`list(ckpt["model"])[:3]`. And "rank X reached line Y" around anything that blocks.

## Part 2 — extend (20 min)

1. **Resume.** `--resume path/to/ckpt.pt` exists. `python check.py --resume` runs 2 epochs,
   resumes for 2 more, and must match a straight 4-epoch run. It doesn't. There is one
   mistake with two separate consequences; find both.
2. **Exact validation.** After part 1, `python check.py --exact` fails for 4 ranks but
   passes for 2. Explain why, then make the accuracy exact for any world size
   (770 val samples).
3. **Gradient accumulation without redundant communication.** `--grad_accum_steps` is
   already numerically correct (`python check.py --accum` passes) but look at the
   "grad all-reduces" count it prints. Make it one all-reduce per *optimizer* step
   (`DDP.no_sync`). The check must still pass and the count must not scale with the
   number of micro-batches. Read the count carefully after your first attempt.

## Stretch (no timer)

4. Swap `LayerNorm` for `BatchNorm1d` in `Block` and run `check.py`. Explain the result.
   Then, on this machine's GPUs: `torchrun --nproc_per_node 2 train.py --device cuda` with
   `nn.SyncBatchNorm.convert_sync_batchnorm`. Does the invariant come back?
5. Which lines in this codebase are wrong on **two nodes** (`torchrun --nnodes 2`)?
   Think `LOCAL_RANK` vs `RANK`, who writes the checkpoint, where `--resume` reads from.
6. Make rank 0 `time.sleep(30)` before the barrier in `fit()` with `--timeout_s 20`.
   What happens, and what are the two ways to fix it?
7. Gather every rank's predictions on the val set to rank 0 and compute a per-class
   accuracy there. Shards are uneven — `all_gather` needs equal shapes. Two ways to
   deal with it.
8. Write the ring all-reduce (reduce-scatter + all-gather with `dist.send/recv`) and
   install it as the comm hook instead of `_counting_allreduce`. `check.py` must pass.

## Reset

Commit `practice2/` before you start (`git add practice2 && git commit -m "tinyddp exercise"`);
`git checkout -- tinyddp train.py` restores the broken version any time.
`patch -p1 < solutions/solution.patch` applies the reference solution (parts 1 + 2).
