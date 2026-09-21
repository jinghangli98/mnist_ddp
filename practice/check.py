"""Objective check: data parallel training must reproduce single-process training.

With the same *global* batch size, N ranks should end up with (numerically)
the same weights as 1 rank. Runs train.py as a subprocess for each setup and
compares the final checkpoints against the world_size=1 reference.

    python check.py            # part 1
    python check.py --accum    # part 2 (gradient accumulation)
"""
import json
import subprocess
import sys

import torch

TOL = 1e-3
TIMEOUT_S = 240


def run(name, **overrides):
    run_dir = f"runs/check_{name}"
    cmd = [sys.executable, "train.py", "--run_dir", run_dir]
    for k, v in overrides.items():
        cmd += [f"--{k}", str(v)]
    print(f"\n>>> {' '.join(cmd[1:])}")
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT after {TIMEOUT_S}s (hang?)")
        return None
    if out.returncode != 0:
        print("    CRASHED. Last lines of stderr:")
        print("    " + "\n    ".join(out.stderr.strip().splitlines()[-5:]))
        return None
    ckpt = torch.load(f"{run_dir}/ckpt.pt")
    metrics = json.load(open(f"{run_dir}/metrics.json"))
    print(f"    final loss {metrics['loss'][-1]:.4f} | val acc {metrics['val_acc']:.4f} "
          f"| {len(metrics['loss'])} optimizer steps")
    return ckpt["model"], metrics


def compare(ref, other):
    if other is None:
        return False
    worst = max((ref[0][k] - other[0][k]).abs().max().item() for k in ref[0])
    same_steps = len(ref[1]["loss"]) == len(other[1]["loss"])
    ok = worst < TOL and same_steps
    print(f"    max |param diff| vs reference: {worst:.2e}  "
          f"{'' if same_steps else '(different number of optimizer steps!) '}"
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    ref = run("ref", world_size=1, batch_size=32)
    assert ref is not None, "single-process reference run failed"

    if "--accum" in sys.argv:
        setups = {"ws2_accum2": dict(world_size=2, batch_size=8, grad_accum_steps=2),
                  "ws1_accum4": dict(world_size=1, batch_size=8, grad_accum_steps=4)}
    else:
        setups = {"ws4": dict(world_size=4, batch_size=8),
                  "ws2": dict(world_size=2, batch_size=16)}

    results = {name: compare(ref, run(name, **kw)) for name, kw in setups.items()}
    print("\n" + "\n".join(f"{'PASS' if ok else 'FAIL'}  {name}" for name, ok in results.items()))
    sys.exit(0 if all(results.values()) else 1)
