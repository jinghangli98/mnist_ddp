"""Objective checks: multi-process training must reproduce single-process training.

With the same *global* batch size, N ranks should end up with (numerically) the same
weights as 1 rank. Runs train.py as a subprocess (via torchrun when N > 1) for each
setup and compares final checkpoints against the world_size=1 reference.

    python check.py            # part 1: 4 ranks x 8 and 2 ranks x 16 vs 1 rank x 32
    python check.py --resume   # part 2.1: 2 epochs + resume 2 more == straight 4 epochs
    python check.py --exact    # part 2.2: val accuracy must match the 1-rank run exactly
    python check.py --accum    # part 2.3: gradient accumulation; prints comm counts
"""
import json
import os
import re
import subprocess
import sys

import torch

TOL = 1e-3
TIMEOUT_S = 240


def run(name, world_size=1, **overrides):
    run_dir = f"runs/check_{name}"
    if world_size == 1:
        cmd = [sys.executable, "train.py"]
    else:
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               "--nproc_per_node", str(world_size), "train.py"]
    cmd += ["--run_dir", run_dir]
    for k, v in overrides.items():
        cmd += [f"--{k}", str(v)]
    shown = " ".join(cmd[1:]).replace("-m torch.distributed.run", "torchrun")
    print(f"\n>>> [world_size {world_size}] {shown}")
    env = dict(os.environ, OMP_NUM_THREADS="1")
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S, env=env)
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT after {TIMEOUT_S}s (hang?)")
        return None
    if out.returncode != 0:
        lines = [l for l in out.stderr.strip().splitlines() if "Warning" not in l]
        excs = sorted({l[:200] for l in lines if re.match(r"^\[rank\d+\]: \S*(Error|Exception)", l)})
        print("    CRASHED. " + ("Exceptions:" if excs else "Last lines of stderr:"))
        print("    " + "\n    ".join(excs or lines[-8:]))
        return None
    ckpt = torch.load(f"{run_dir}/ckpt.pt", map_location="cpu")
    metrics = json.load(open(f"{run_dir}/metrics.json"))
    print(f"    final loss {metrics['loss'][-1]:.4f} | val acc {metrics['val_acc']:.4f} | "
          f"{len(metrics['loss'])} optimizer steps | stopped after epoch {metrics['epoch']} | "
          f"grad all-reduces {metrics.get('comm_calls', '?')}")
    return ckpt["model"], metrics


def compare(ref, other, exact_acc=False):
    if other is None:
        return False
    ref_sd, other_sd = ref[0], other[0]
    if set(ref_sd) != set(other_sd):
        extra = sorted(set(other_sd) - set(ref_sd))[:3]
        missing = sorted(set(ref_sd) - set(other_sd))[:3]
        print(f"    state_dict keys differ from reference -> FAIL\n"
              f"      unexpected: {extra}...\n      missing: {missing}...")
        return False
    worst = max((ref_sd[k] - other_sd[k]).abs().max().item() for k in ref_sd)
    same_steps = len(ref[1]["loss"]) == len(other[1]["loss"])
    acc_diff = abs(ref[1]["val_acc"] - other[1]["val_acc"])
    ok = worst < TOL and same_steps and (not exact_acc or acc_diff < 1e-9)
    print(f"    max |param diff| vs reference: {worst:.2e}  |val acc diff| {acc_diff:.2e}  "
          f"{'' if same_steps else '(different number of optimizer steps!) '}"
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--part1"
    results = {}

    if mode == "--resume":
        no_stop = dict(target_acc=1.0)  # early stopping off, so every run is 4 epochs
        ref = run("straight4", world_size=1, batch_size=32, epochs=4, **no_stop)
        assert ref is not None, "single-process reference run failed"
        first = run("first2", world_size=4, epochs=2, **no_stop)
        if first is None:
            results["resume"] = False
        else:
            resumed = run("resumed", world_size=4, epochs=4, resume="runs/check_first2/ckpt.pt",
                          **no_stop)
            results["resume (ws4: 2 epochs + 2 resumed == ws1: 4 epochs)"] = compare(ref, resumed)
    else:
        ref = run("ref", world_size=1, batch_size=32)
        assert ref is not None, "single-process reference run failed"
        if mode == "--accum":
            setups = {"ws2_accum2": dict(world_size=2, batch_size=8, grad_accum_steps=2),
                      "ws4_accum4": dict(world_size=4, batch_size=2, grad_accum_steps=4)}
        else:  # part 1, and --exact (same runs, val accuracy must match to the last digit)
            setups = {"ws4": dict(world_size=4, batch_size=8),
                      "ws2": dict(world_size=2, batch_size=16)}
        for name, kw in setups.items():
            results[name] = compare(ref, run(name, **kw), exact_acc=(mode == "--exact"))

    print("\n" + "\n".join(f"{'PASS' if ok else 'FAIL'}  {name}" for name, ok in results.items()))
    sys.exit(0 if all(results.values()) else 1)
