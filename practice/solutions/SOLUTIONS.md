# SPOILERS — finish the exercise first

Apply everything: `cd practice && patch -p1 < solutions/solution.patch`

## Part 1: the five bugs

Ordered roughly by how you'd hit them.

### 1. Loss explodes → gradients are summed, not averaged (`parallel.py: sync_grads`)
`all_reduce(SUM)` with no division by `world_size`. Each rank's loss is already a mean
over its local batch, so the sum is `world_size ×` the true gradient ⇒ effective LR ×4 ⇒
divergence. **How to show it:** print grad norm before/after sync — it grows ~×4 instead
of staying the same magnitude. Fix: `p.grad.div_(world_size)` (or `ReduceOp.AVG` on NCCL;
gloo doesn't support AVG — worth knowing).
Side note: this is why the code uses SGD. With Adam the update is scale-invariant and
this bug would be nearly invisible.

### 2. Hang ~60 s then "Connection closed by peer" → collective inside `if is_main()` (`trainer.py: fit`)
`dist.barrier()` is only called by rank 0. The other ranks skip it and their next
collective is the first grad `all_reduce` of the next epoch, so the collectives are
mismatched; after the last epoch the others simply exit and rank 0's barrier dies.
**Rule: every rank must call the same collectives in the same order.** How to show it:
`print(f"[rank {r}] before barrier", flush=True)` — only rank 0 prints. Fix: dedent the barrier.

### 3. Replicas start from different weights (`train.py` + `parallel.py`)
`torch.manual_seed(cfg.seed + rank)` runs before `build_model`, and the wrapper never
broadcasts. Averaged gradients applied to different weights ⇒ replicas never agree; the
checkpoint is just rank 0's opinion. **How to show it:** print a param checksum per rank
at step 0 — four different numbers. Fix: broadcast params *and buffers* from rank 0 in
`DataParallel.__init__` (this is what real DDP does). Seeding identically also works but
is fragile; per-rank seeds are legitimately wanted for dropout/augmentation.

### 4. Shards overlap (`data.py: _indices`)
The permutation is seeded with `seed + epoch + rank`, so every rank shuffles
*differently* and then strides `[rank::world_size]` through its own private permutation.
The shards are no longer a partition: some samples seen by several ranks, some by none.
Training still "works", which is what makes it nasty. **How to show it:** gather/print each
rank's indices for one epoch; `len(set(all)) < n`. Fix: same permutation everywhere
(`seed + epoch`), rank only picks the stride offset. This is exactly what
`DistributedSampler` does.

### 5. Same data order every epoch (`trainer.py: fit`) — not caught by check.py
`ShardedLoader.set_epoch` exists but nobody calls it (the reference run has the same
flaw, so the check can't see it). Fix: `self.train_loader.set_epoch(epoch)`.
You already know this one from `DistributedSampler.set_epoch`.

## Part 2

**Global eval.** Accumulate `correct`/`total` as a tensor and do ONE `all_reduce` after
the loop. Trap: 770 samples over 4 ranks → shards of 193/193/192/192 → with batch 64 ranks
0–1 have 4 batches and ranks 2–3 have 3. An all-reduce *inside* the loop is called a
different number of times per rank → hang. Summing counts (not averaging per-rank
accuracies) keeps it exact with uneven shards. Contrast with `DistributedSampler`, which
pads by duplicating samples, so your MNIST test accuracy is very slightly off.

**Grad accumulation.** Divide the loss by `accum`, `backward()` every micro-batch, but
`sync_grads()` + `step()` + `zero_grad()` only on the last one. Since sync is an explicit
call here, skipping it is trivial; with real DDP the same thing is `with model.no_sync():`
on the non-final micro-batches. Move `zero_grad` out of the per-micro-batch path or you
wipe what you accumulated.

## Things to be able to say out loud

- DDP = replicate model, shard data, all-reduce(mean) grads ⇒ mathematically the same as
  one big batch. Global batch = per-rank batch × world size × accum; LR is often scaled with it.
- Real DDP: broadcasts params/buffers at construction; autograd hooks fire as grads become
  ready; grads are bucketed (25 MB default) in reverse order and all-reduced
  asynchronously so comm overlaps with backward; `find_unused_parameters` exists because a
  param with no grad never fires its hook and the bucket would wait forever.
- Ring all-reduce = reduce-scatter + all-gather; each rank sends `2(N-1)/N × size` — bandwidth
  cost independent of N.
- Collectives: `broadcast`, `all_reduce`, `reduce`, `all_gather`, `reduce_scatter`,
  `gather/scatter`, `barrier`; p2p `send/recv`. `rank` vs `local_rank` vs `world_size`.
- Hang checklist: collective under a rank conditional; uneven number of batches; an
  exception on one rank; mismatched tensor shapes/dtypes; rank 0-only work that takes
  longer than the timeout.
- Rank 0 only: logging, checkpoint writes. `model.module.state_dict()`; `map_location` on load.
- BatchNorm stats are per-rank unless `SyncBatchNorm`.
- Where DDP stops: model must fit on one GPU → ZeRO/FSDP (shard optimizer state, grads,
  params), tensor parallel, pipeline parallel.
