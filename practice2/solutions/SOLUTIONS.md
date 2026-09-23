# SPOILERS — finish the exercise first

Apply everything: `cd practice2 && patch -p1 < solutions/solution.patch`

## Part 0 in one paragraph

`torchrun` starts N processes and gives each one `RANK`, `LOCAL_RANK`, `WORLD_SIZE`,
`MASTER_ADDR`, `MASTER_PORT`; `init_process_group` reads them. Without those variables
`setup()` returns early and the script is an ordinary single-process trainer — no
`DDP`, no sampler sharding, no collectives. `DDP(model)` does three things: at
construction it broadcasts rank 0's parameters and buffers; in `forward` it re-broadcasts
buffers (BatchNorm running stats) and decides whether the coming backward will reduce;
in `backward`, autograd hooks fire as each parameter's gradient becomes ready, grads are
copied into buckets, and each full bucket is all-reduced (mean) asynchronously — that is
the call `_counting_allreduce` intercepts. Explicit collectives in the code:
`all_reduce` of the running loss (logging), `reduce` in `evaluate`, `barrier` after
the checkpoint. Global batch = 8 × 4 × 1 = 32, so 4096 / 32 = 128 steps/epoch.
Rank 0 only: `log`, `save_checkpoint`. Everything else runs everywhere.

## What you'd see, in order

1. `torchrun ... train.py`: "512 batches/epoch/rank" (should be 128), then at step 2 a
   crash: `Expected to have finished reduction in the prior iteration...`. → bug 1.
2. With that patched: two epochs run, rank 0 prints "target accuracy reached, stopping",
   then ranks 1–3 die in `loss.backward()` with `Connection closed by peer`. → bug 4.
3. With that patched, `check.py`: "different number of optimizer steps" and huge param
   diff → bug 2; after it, params still off by ~1e-2 → bug 3; after it, "state_dict keys
   differ" → bug 5.

## Part 1: the five bugs

### 1. Crash on 2nd iteration → parameter that never gets a gradient (`model.py` + `trainer.py`)
`aux_head` is always built and run in `forward`, but its loss is only added when
`aux_weight > 0` (default 0). Its grad hook never fires, so the bucket it lives in never
fills and the reducer is still waiting when the next `forward` starts. Single-process has no
reducer, so it's fine there. **Show it:** `TORCH_DISTRIBUTED_DEBUG=DETAIL` names the
parameter, or `[p for n, p in model.named_parameters() if p.grad is None]` after backward.
**Fixes,** best first: don't build what you don't use (patch); or always add
`aux_weight * aux_loss` even when the weight is 0 (autograd then produces a zero grad);
or `find_unused_parameters=True` — works, but DDP then walks the autograd graph every
iteration to find unused params, and it's the wrong answer if the param *should* train.
Worth knowing: parameters that are used on *some* ranks but not others hang instead of
crashing (bucket ready on one rank, not on another) — that is the case
`find_unused_parameters` is really for.

### 2. Silent, 4× more steps → training loader isn't sharded (`data.py`)
`DataLoader(train_ds, shuffle=True)` gives every rank the *whole* dataset. Worse, all ranks
seeded `torch.manual_seed(0)` identically, so every rank draws the *same* permutation:
all four processes compute the same gradient and the all-reduce averages four identical
things. It is single-process training that costs 4 machines. **Show it:** print
`len(train_loader)` (512, not 128) and `y[:8].tolist()` for the first batch on every rank
(identical). **Fix:** `sampler=shard(train_ds, shuffle=True)` and drop `shuffle=True`
(they're mutually exclusive). `set_epoch` was already wired up and starts working the
moment the sampler is a `DistributedSampler`.

### 3. Silent, trains ~4× slower → loss summed and divided by the *global* batch (`trainer.py`)
`cross_entropy(reduction="sum") / global_batch_size` gives each rank
`sum over 8 samples / 32` = ¼ of its local mean. DDP then **averages** across ranks:
¼ × mean of local means = ¼ of the true global-batch gradient. Effective LR ÷ world size.
On 1 process `global_batch = 32 = local batch`, so it's exactly the mean and looks right.
**Show it:** total gradient norm after `backward()` at step 0, 1 rank vs 4 ranks — ×¼.
**Fix:** per-rank mean (`reduction="mean"`), i.e. give DDP what it expects: a loss that is
already a mean over local samples. Rule: DDP averages, so your per-rank loss must be
normalised per rank, not globally. (If you ever *want* a sum, use `all_reduce` on the
grads yourself, or a comm hook without the `/ world_size`.)

### 4. Hang then "Connection closed by peer" → `reduce` feeding control flow (`trainer.py`)
`dist.reduce(correct, dst=0)` only delivers the total to rank 0; on the other ranks the
tensor is untouched (gloo) or unspecified (in general). `correct / len(dataset)` then gives
the right number on rank 0 and ~¼ of it elsewhere. `if val_acc >= target: break` is
therefore taken by rank 0 only: it saves, destroys the process group and exits, the others
start the next epoch, and their first grad all-reduce finds the peer gone. **Show it:**
`print(f"[rank {r}] val_acc {val_acc}")` on every rank. **Fix:** `all_reduce` — every rank
needs the value because every rank branches on it. Also fixed here: reduce
`[correct, total]` and divide by the reduced total instead of `len(dataset)`
(the sampler pads to 772, see part 2.2). Rule from set 1 again, generalised: **control flow
must be identical on all ranks**. A decision made from data that lives only on rank 0
must be `broadcast` (a 1-element tensor) before anyone acts on it.

### 5. Checkpoint from N ranks won't load into the single-process model (`trainer.py`)
`self.model.state_dict()` on the DDP wrapper prefixes every key with `module.`.
`check.py` reports "state_dict keys differ"; in real life the inference script fails with
`Missing key(s) ... Unexpected key(s) "module.embed.weight"`. **Fix:** save
`unwrap(model).state_dict()` (`model.module` if wrapped). While here: `torch.load` should
get `map_location=device` on load, or every rank on a multi-GPU node deserialises onto
`cuda:0` (the device the tensors were saved from).

## Part 2

### 2.1 Resume: `if cfg.resume and is_main():` (`train.py`)
The comment is a trap: DDP *does* broadcast the weights at construction, so the model
part of this is genuinely fine. What is not broadcast: **(a)** `state["epoch"]` — rank 0
starts at epoch 2, the others at 0. Rank 0 finishes two epochs early, exits, the others
hang in the next all-reduce → "Connection closed by peer". **(b)** the optimizer state —
rank 0 has momentum buffers, the others start from zero, so the same averaged gradient
produces different updates on different ranks and the replicas drift apart silently
(after fixing only (a), `check.py --resume` fails with a diff around 1e-1). **Show (b):**
`sum(b.sum() for s in opt.state.values() for b in s.values())` per rank right after
loading. **Fix:** every rank loads the checkpoint (with `map_location`). Loading on rank 0
and broadcasting is legitimate for huge checkpoints on slow shared filesystems, but then
you must broadcast *everything*: weights, optimizer state (`broadcast_object_list`), epoch.
Also in a real resume: RNG state, sampler epoch (handled here because `set_epoch(epoch)`
uses the real epoch number), LR scheduler.

### 2.2 Exact validation: `DistributedSampler` pads
`DistributedSampler` makes shards equal by *repeating* the first few samples: 770 over 4
ranks → 772, two samples counted twice. Over 2 ranks 770 is already even, so ws2 passes
and ws4 fails by a fraction of a percent. Same reason your MNIST `test()` accuracy in
`train_ddp.py` is not exactly the single-GPU number. **Fix in the patch:** a tiny
`UnpaddedDistributedSampler` (rank r takes `range(r, n, W)`), shards of 193/193/192/192,
one `all_reduce` of `[correct, total]` after the loop (uneven batch counts are fine since
the collective is outside the loop). Alternatives: `all_gather` predictions and drop the
padded tail (`DistributedSampler` puts the duplicates at the end), or evaluate on rank 0
only (simple, but rank 0 then does W× the work while the others wait — see stretch 6).

### 2.3 `no_sync` — and the trap
Wrap the non-final micro-batches in `with model.no_sync():`. **The trap:** `no_sync` sets a
flag that `forward` reads (`require_backward_grad_sync`); the *forward* must be inside the
context, not just `backward()`. If you wrap only the backward, the comm count stays at
`accum` per step and everything else looks fine — that is exactly why the hook counter is
there. Correct count: 256 all-reduces for 256 optimizer steps regardless of `accum`.
Why it's numerically identical either way: all-reduce is linear, so averaging every
micro-batch's grad and summing equals summing locally and averaging once. One bucket here
because the model is ~55k params (< the 1 MB first bucket); a real model has many buckets
and the count is `#buckets × steps`.

## Things to be able to say out loud

- `rank` = global index across all nodes (use for "am I main", checkpoint writes);
  `local_rank` = GPU index on this node (use for `cuda.set_device`); `world_size` = total.
  torchrun: `--nnodes`, `--nproc_per_node`, `--rdzv_endpoint`; `--standalone` for one node.
- DDP construction: broadcast params+buffers from rank 0, build buckets in *reverse*
  parameter order (last layers' grads are ready first). Backward: per-param hook →
  bucket → async all-reduce, overlapping with the rest of backward. Forward: buffer
  broadcast, `prepare_for_backward`. `no_sync`, `find_unused_parameters`, `static_graph`,
  `bucket_cap_mb`, `gradient_as_bucket_view`, `register_comm_hook` (fp16 compression,
  PowerSGD, or your own ring).
- DDP averages gradients ⇒ per-rank loss must be a per-rank mean. Global batch =
  per-rank batch × world size × accum. Linear LR scaling rule + warmup when you grow it.
- `reduce` vs `all_reduce`; `gather`/`all_gather` need equal shapes (pad, or
  `all_gather_object`); `broadcast_object_list` for Python objects; `barrier` is not a
  synchronisation of *data*, only of *time*.
- Hang checklist (now longer): collective under a rank conditional; control flow decided
  from rank-local data (metrics, early stopping, `if loss.isnan(): break`); unequal
  number of batches / collectives per rank; parameter unused on some ranks; one rank
  raising an exception; rank 0-only work longer than the timeout; mismatched shapes/dtypes.
  Tools: `TORCH_DISTRIBUTED_DEBUG=DETAIL`, `TORCH_CPP_LOG_LEVEL=INFO`, `py-spy dump` on a
  hung rank, `NCCL_DEBUG=INFO`, `torch.distributed.monitored_barrier` (gloo only).
- Checkpoints: `model.module.state_dict()`, `map_location`, save from rank 0 then
  `barrier`, every rank loads. Resume = weights + optimizer + scheduler + epoch + RNG +
  sampler epoch.
- Dropout / augmentation: seed *per rank* (`seed + rank`) or every replica drops the
  same units; model init: identical seed, or rely on DDP's broadcast. BatchNorm: per-rank
  statistics unless `SyncBatchNorm` (GPU only).
- Beyond DDP: memory is replicated W×. ZeRO stages 1/2/3 shard optimizer state / grads /
  params (FSDP); tensor parallel splits matmuls; pipeline parallel splits layers.
