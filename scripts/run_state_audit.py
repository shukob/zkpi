#!/usr/bin/env python3
"""What the state-update audit costs, and that it still rejects at scale.

The policy audit answers "is this rule well-formed". This answers "did the book
that rule drives actually move the way it said", which is the half RFS needs and
the half that was missing. Cost matters because it is paid per fill, not per
policy: a maker quoting a stream pays this on every step of the chain, and the
venue pays the verification on every step of every maker.

The rejection arms are measured alongside rather than left to the unit tests,
because a cost table for a check that has stopped checking is worse than no
table at all.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.hosts import this_host
from scripts.measure import exact, render, summarise, value    # noqa: E402                                          # noqa: E402
from zk.groups import make_group                                             # noqa: E402
from zk.state_audit import StateAuditor                                      # noqa: E402


def calibration(group, repeats: int = 200) -> dict:
    point = group.hash_to_point(b"calibration")
    samples = []
    for _ in range(repeats):
        t = time.perf_counter()
        group.point_pow(point, 12345)
        samples.append((time.perf_counter() - t) * 1e6)
    return {"scalar_mult_us": summarise(samples)}


def one_chain(auditor, length: int, limit_value: int):
    """An honest run, timed on both sides."""
    limit_blinding = auditor.key.random_blinding()
    t = time.perf_counter()
    limit = auditor.commit_limit(limit_value, limit_blinding)
    limit_ms = (time.perf_counter() - t) * 1e3

    blinding = auditor.key.random_blinding()
    opening = auditor.key.commit(0, blinding)
    inventory, steps, prove_ms = 0, [], []
    swing = max(1, limit_value // 8)
    for index in range(length):
        filled = swing if index % 2 else -swing
        new_blinding = auditor.key.random_blinding()
        t = time.perf_counter()
        step, inventory = auditor.prove_update(
            step=index, old_inventory=inventory, old_blinding=blinding,
            filled=filled, fill_blinding=auditor.key.random_blinding(),
            limit=limit_value, limit_blinding=limit_blinding,
            new_blinding=new_blinding)
        prove_ms.append((time.perf_counter() - t) * 1e3)
        steps.append(step)
        blinding = new_blinding

    t = time.perf_counter()
    ok, reason = auditor.verify_chain(opening, steps, limit)
    verify_ms = (time.perf_counter() - t) * 1e3
    return {"steps": length, "limit_commit_ms": limit_ms,
            # One sample per step, so the chain's own length is the sample count.
            "prove_per_step": summarise(prove_ms),
            "verify_total_ms": verify_ms, "verify_ms_per_step": verify_ms / length,
            "accepted": ok, "reason": reason,
            # Wire size is a function of the construction, not of the machine.
            "step_bytes": exact(_step_bytes(auditor, steps[0]))}, (opening, steps, limit)


def _step_bytes(auditor, step) -> int:
    """Wire size of one link: the points plus the two range proofs."""
    group = auditor.group
    points = sum(len(group.encode(p)) for p in (step.inventory, step.fill))
    scalars = 32 * 2                     # the arithmetic proof's opening
    ranges = 0
    for proof in (step.below_cap, step.above_floor):
        ranges += proof.bits * (len(group.encode(step.inventory)) * 2 + 32 * 3)
    return points + scalars + ranges + len(step.follows)


def rejections(auditor, limit_value: int) -> list[dict]:
    """Every way the chain is supposed to break, exercised against the clock."""
    out = []

    # a state that does not follow the one before it
    _, (opening, steps, limit) = one_chain(auditor, 3, limit_value)
    t = time.perf_counter()
    ok, reason = auditor.verify_chain(opening, [steps[0], steps[2]], limit)
    out.append({"attack": "replayed or skipped state", "accepted": ok,
                "reason": reason, "ms": (time.perf_counter() - t) * 1e3})

    # two books run in parallel
    _, (other_opening, other_steps, _) = one_chain(auditor, 3, limit_value)
    t = time.perf_counter()
    ok, reason = auditor.verify_chain(opening, [steps[0], other_steps[1]], limit)
    out.append({"attack": "forked state", "accepted": ok, "reason": reason,
                "ms": (time.perf_counter() - t) * 1e3})

    # the promise swapped for a looser one after the fact
    # loose enough to be a different promise, still under the public ceiling
    looser = auditor.commit_limit(min(limit_value * 2, auditor.ceiling),
                                  auditor.key.random_blinding())
    t = time.perf_counter()
    ok, reason = auditor.verify_chain(opening, steps, looser)
    out.append({"attack": "limit swapped for a looser one", "accepted": ok,
                "reason": reason, "ms": (time.perf_counter() - t) * 1e3})

    # a maker past its own limit cannot even build the step
    blinding = auditor.key.random_blinding()
    try:
        auditor.prove_update(
            step=0, old_inventory=limit_value - 1, old_blinding=blinding,
            filled=-limit_value, fill_blinding=auditor.key.random_blinding(),
            limit=limit_value, limit_blinding=auditor.key.random_blinding(),
            new_blinding=auditor.key.random_blinding())
        out.append({"attack": "breach the committed limit", "accepted": True,
                    "reason": "the prover built a step it should not have", "ms": 0.0})
    except ValueError as exc:
        out.append({"attack": "breach the committed limit", "accepted": False,
                    "reason": str(exc), "ms": 0.0})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("artifacts/state_audit.json"))
    ap.add_argument("--lengths", type=int, nargs="+", default=[1, 5, 20, 50])
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--ceiling", type=int, default=1 << 13)
    args = ap.parse_args()

    group = make_group("ed25519")
    auditor = StateAuditor(group, ceiling=args.ceiling)
    result = {"host": this_host(), "python": platform.python_version(),
              "group": "ed25519", "ceiling": args.ceiling, "limit": args.limit,
              "calibration": calibration(group)}
    print(f"calibration: scalar mult {value(value(result['calibration']['scalar_mult_us'])):.1f} us",
          flush=True)

    chains = []
    for length in args.lengths:
        row, _ = one_chain(auditor, length, args.limit)
        chains.append(row)
        print(f"  {length:3} steps  prove {render(row['prove_per_step'], 1)} ms/step  "
              f"verify {row['verify_ms_per_step']:7.1f} ms/step  "
              f"{row['step_bytes']['exact']:,} B/step  accepted={row['accepted']}", flush=True)
    result["chains"] = chains
    result["rejections"] = rejections(auditor, args.limit)
    for row in result["rejections"]:
        print(f"  {row['attack']:34} accepted={row['accepted']}", flush=True)
    if any(r["accepted"] for r in result["rejections"]):
        print("A REJECTION ARM ACCEPTED. The audit is not checking.", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
