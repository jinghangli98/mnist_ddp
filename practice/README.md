# Mock interview: `minidp`

A small research codebase that implements data-parallel training by hand on top of
`torch.distributed` (no `DistributedDataParallel`). Runs on CPU with the `gloo`
backend, like you'd get in CoderPad/Colab. A full run takes a few seconds.

**Rules (same as the real thing):** 45-minute timer, no AI tools, PyTorch docs allowed,
talk out loud, add lots of `print()`s. Don't open `solutions/` until you're done.

```bash
cd practice
python train.py --world_size 1 --batch_size 32   # single-process reference: works, ~77% val acc
python train.py                                  # 4 ranks: ...does not go well
python check.py                                  # objective pass/fail for part 1
```

## Part 0 — read (5 min)

Before touching anything, be able to say out loud: what does each file do, where
does communication happen, what is the global batch size, and what runs on every
rank vs. only on rank 0?

## Part 1 — debug (20 min)

A colleague says "it trains fine on 1 process but multi-process is broken".
The invariant the code must satisfy:

> With the same global batch size, training on N ranks yields the same weights as
> training on 1 rank (up to float noise).

`python check.py` tests exactly this and must print `PASS` for both setups.
There are several independent bugs. Not all of them are caught by `check.py` —
one you can only find by reading. For each one, *show* it with a print before fixing it.

Useful things to print (from every rank, with the rank in the message, `flush=True`):
a checksum of the parameters (`sum(p.sum() for p in model.parameters())`), a gradient
norm before/after sync, the first few sample indices each rank draws per epoch,
"rank X reached line Y" around anything that blocks.

## Part 2 — extend (20 min)

1. **Global eval.** `evaluate()` reports accuracy of rank 0's shard only. Make it the
   exact accuracy over the whole validation set (770 samples — note it isn't divisible
   by the world size). Also log the global mean training loss instead of rank 0's.
2. **Gradient accumulation.** `--grad_accum_steps` exists in the config but is ignored.
   Implement it, communicating only once per optimizer step.
   `python check.py --accum` must pass.

## Stretch (no timer)

3. `sync_grads` does one all-reduce per parameter tensor. Flatten into a single buffer
   (or a few buckets), all-reduce once, copy back. Time both with `--hidden 512 --depth 8`.
4. Implement `all_reduce` yourself as a ring (reduce-scatter + all-gather) with
   `dist.send/recv` or `dist.batch_isend_irecv`, and swap it in. `check.py` must still pass.
5. Make communication overlap with backward using
   `p.register_post_accumulate_grad_hook` + `async_op=True`, waiting on the handles
   before `opt.step()`. How does this interact with gradient accumulation?
6. Add `--resume`: restore model, optimizer (momentum!), epoch. A run of 2 epochs +
   resume for 2 more must match a straight 4-epoch run.
7. Port it to `torchrun` (env-var rendezvous) instead of `mp.spawn`.

## Reset

Commit `practice/` before you start (`git add practice && git commit -m "minidp exercise"`);
then `git checkout -- minidp train.py` restores the broken version any time.
`patch -p1 < solutions/solution.patch` applies the reference solution (parts 1 + 2).
