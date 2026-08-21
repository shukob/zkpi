#!/usr/bin/env python3
"""What public verifiability costs when it is bought without a group.

`input_check.json` measured the *designated-verifier* VOLE commitment against
Pedersen and found 113x on one scale. That number is real and it is misleading,
because only the holder of `Delta` can check a designated-verifier opening, and
a regulator who was not present is exactly the party that has to check. This
runner measures the transform that fixes it.

Both arms prove the same statement --- one public linear combination of a
committed vector --- and both are publicly verifiable, so the comparison is
between two ways of doing the same job rather than between a proof and half a
proof. `pedersen` is the stack's input check on ed25519. `voleith` is
VOLE-in-the-Head over a 127-bit prime: a GGM tree per repetition, all-but-one
opening, Fiat--Shamir for `Delta`, no group anywhere.

Three things come out, and only two of them transfer between machines.

**Proof bytes and hash counts are exact** and mean the same thing on any
hardware, which is why they are reported by part rather than as a total.

**Wall clock is not a like-for-like comparison and the artifact says so.**
Pedersen runs on libsodium through PyNaCl, which is C. VOLEitH runs on
`hashlib` --- also C --- with the field arithmetic in CPython. The ratio is
therefore an upper bound on how much slower the transform is, not an estimate of
it, and `--phases` splits the VOLEitH time into the part that is hashing and the
part that is Python so the bound can be read rather than guessed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.hosts import this_host                                  # noqa: E402
from scripts.measure import exact, render, summarise                 # noqa: E402
from zk import voleith                                               # noqa: E402
from zk.scheme import make_linear_proof                              # noqa: E402

P = (1 << 127) - 1
MASK_BITS = 119                # what input_check's mask is at 166 inputs
VALUE_BITS = 31


def sample(n: int) -> list[int]:
    """166 prices and a mask, which is the shape the input check actually has."""
    return [secrets.randbelow(1 << VALUE_BITS) for _ in range(n - 1)] + [
        secrets.randbelow(1 << MASK_BITS)]


def time_arm(scheme, values, context: bytes, repeats: int) -> dict:
    proof = scheme.prove_linear(values, context)
    accepted, why = scheme.verify_linear(proof, context)
    build, check = [], []
    for _ in range(repeats):
        t0 = time.perf_counter()
        proof = scheme.prove_linear(values, context)
        t1 = time.perf_counter()
        scheme.verify_linear(proof, context)
        t2 = time.perf_counter()
        build.append((t1 - t0) * 1e3)
        check.append((t2 - t1) * 1e3)
    return {"accepted": accepted, "why": why,
            "prove_ms": summarise(build), "verify_ms": summarise(check),
            "bytes": exact(scheme.proof_bytes(proof)),
            "size_breakdown": scheme.size_breakdown(proof)}


def phase_split(n: int, depth: int, repeats: int, rounds: int) -> dict:
    """How much of the VOLEitH time is hashing and how much is CPython.

    The hashing is C and would stay about where it is in a compiled
    implementation; the rest is the part a rewrite would take away. Measured by
    running the leaf expansion on its own, at exactly the volume a proof does.
    """
    pack = voleith.Packing(P, n, depth)
    seeds = [secrets.token_bytes(voleith.SEED_BYTES) for _ in range(1 << depth)]
    leaves = repeats * (1 << depth)

    xof = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(repeats):
            for i, seed in enumerate(seeds):
                hashlib.shake_128(seed + i.to_bytes(4, "big")).digest(pack.blob_bytes)
        xof.append((time.perf_counter() - t0) * 1e3)

    mask = pack.mask
    both = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(repeats):
            acc = weighted = 0
            for i, seed in enumerate(seeds):
                t = pack.leaf(seed, 0, i, mask)
                acc += t
                weighted += i * t
        both.append((time.perf_counter() - t0) * 1e3)

    return {"leaves": exact(leaves),
            "prg_bytes": exact(leaves * pack.blob_bytes),
            "xof_only_ms": summarise(xof), "xof_and_packing_ms": summarise(both)}


def parameter_sweep(n: int) -> list[dict]:
    """The size-against-computation trade, which FAEST's table 2 also shows.

    A deeper tree means fewer repetitions for the same soundness, so fewer
    consistency corrections and a smaller proof --- bought with `2^depth` PRG
    calls per repetition. Reproducing the shape of a published table on this
    stack's own statement is the check that the harness is measuring the thing.
    """
    out = []
    for depth in (4, 6, 8, 10, 12, 14, 16):
        repeats = -(-128 // depth)
        parts = voleith.proof_size(n, P.bit_length(), depth, repeats)
        out.append({"depth": depth, "leaves_per_tree": 1 << depth,
                    "repeats": repeats, "soundness_bits": parts["soundness_bits"],
                    "bytes": parts["total"], "hashes": parts["hashes"]})
    return out


def linear_code_arithmetic(n: int, depth: int = 8) -> dict:
    """What section 6.1 of the paper would buy, computed rather than measured.

    The `[tau,1,tau]` repetition code costs `(tau-1)*n` field elements of
    consistency correction, and over a 127-bit prime that is the whole proof. A
    linear code of distance `d_C` replaces it with `ceil(n/k_C)*(n_C-k_C)`,
    bought with `n_C` trees instead of `tau`. Singleton bounds `d_C` by
    `n_C-k_C+1`, so a Reed--Solomon code is the best case and this is its
    arithmetic. **Not implemented**: it is here so the gap between what was
    measured and what the construction can reach is a number.
    """
    width = (P.bit_length() + 7) // 8
    distance = -(-128 // depth)              # |S_Delta|^-d_C <= 2^-128
    best = None
    for k_c in (4, 8, 16, 32, 64, 128):
        n_c = k_c + distance - 1             # MDS
        rows = -(-n // k_c)
        total = (voleith.COMMIT_BYTES + n * width
                 + rows * (n_c - k_c) * width
                 + n_c * depth * voleith.SEED_BYTES
                 + n_c * voleith.COMMIT_BYTES + n_c * width + width)
        row = {"k_C": k_c, "n_C": n_c, "d_C": distance, "rows": rows,
               "bytes": total, "hashes": n_c * (1 << depth) * 2}
        if best is None or total < best["bytes"]:
            best = row
    return {"note": "arithmetic only, not implemented", "depth": depth,
            "best": best}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=int, nargs="+", default=[16, 64, 167])
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--depth", type=int, default=voleith.DEFAULT_DEPTH)
    ap.add_argument("--tree-repeats", type=int, default=voleith.DEFAULT_REPEATS)
    ap.add_argument("--phases", action="store_true",
                    help="split the VOLEitH time into hashing and CPython")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "artifacts" / "voleith.json")
    args = ap.parse_args()

    schemes = {
        "pedersen": make_linear_proof("pedersen", value_bits=VALUE_BITS),
        "voleith": make_linear_proof("voleith", modulus=P, depth=args.depth,
                                     repeats=args.tree_repeats),
    }
    result = {"host": this_host(), "repeats": args.repeats,
              "modulus_bits": P.bit_length(), "depth": args.depth,
              "tree_repeats": args.tree_repeats,
              "soundness_bits": args.depth * args.tree_repeats,
              "caveat": ("Pedersen is libsodium through PyNaCl, which is C. "
                         "VOLEitH is hashlib plus CPython field arithmetic. The "
                         "wall-clock ratio is an upper bound on the transform's "
                         "cost, not an estimate of it. Bytes and hash counts do "
                         "not have this problem."),
              "arms": {}}

    for n in args.inputs:
        values = sample(n)
        context = b"qomm:voleith-bench:" + str(n).encode()
        for name, scheme in schemes.items():
            row = time_arm(scheme, values, context, args.repeats)
            row["n_inputs"] = n
            result["arms"].setdefault(name, []).append(row)
            print(f"{name:>9} {n:>4} inputs  "
                  f"prove {render(row['prove_ms'], 2, ' ms'):>20}  "
                  f"verify {render(row['verify_ms'], 2, ' ms'):>20}  "
                  f"{row['bytes']['exact']:>7} B  accepted={row['accepted']}")

    ped = {r["n_inputs"]: r for r in result["arms"]["pedersen"]}
    vol = {r["n_inputs"]: r for r in result["arms"]["voleith"]}
    result["ratios"] = {
        str(n): {"prove": round(vol[n]["prove_ms"]["median"]
                                / ped[n]["prove_ms"]["median"], 2),
                 "verify": round(vol[n]["verify_ms"]["median"]
                                 / ped[n]["verify_ms"]["median"], 2),
                 "bytes": round(vol[n]["bytes"]["exact"] / ped[n]["bytes"]["exact"], 2)}
        for n in args.inputs}

    result["sweep"] = parameter_sweep(max(args.inputs))
    result["linear_code"] = linear_code_arithmetic(max(args.inputs), args.depth)
    if args.phases:
        result["phases"] = phase_split(max(args.inputs), args.depth,
                                       args.tree_repeats, max(3, args.repeats // 6))

    print("\nVOLEitH over Pedersen, same statement, both publicly verifiable:")
    for n, row in result["ratios"].items():
        print(f"  {n:>5} inputs  prove {row['prove']:>6.2f}x  "
              f"verify {row['verify']:>6.2f}x  bytes {row['bytes']:>6.2f}x")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0 if all(r["accepted"] for rows in result["arms"].values()
                    for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
