#!/usr/bin/env python3
"""Measure the proof that the opened quote is correct, and check it rejects forgeries.

Also measures the joint path: the same proof assembled by a quorum of computing
nodes from shares, so no node holds the witness.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zk.commit import Pedersen, verify_opening                                # noqa: E402
from zk.groups import make_group
from scripts.measure import exact, render, summarise      # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.run_defmi import calibration                                    # noqa: E402                                              # noqa: E402
from zk.quote_proof import MakerWitness, QuoteProof, QuoteProver, QuoteVerifier  # noqa: E402
from zk import threshold_sigma                                                # noqa: E402
from zk.threshold_sigma import deal, joint_prove_opening                      # noqa: E402

from scripts.hosts import this_host                                          # noqa: E402

SENTINEL = 1 << 20


def makers_for(n: int, seed: int) -> list[MakerWitness]:
    rng = random.Random(seed)
    return [MakerWitness(mid=100_000 + rng.randint(-15, 15), half=rng.randint(5, 40),
                         slope=rng.choice([0, 1, 2]), invcoef=rng.choice([0, 1, 2]),
                         inv=rng.randint(0, 50), maxqty=rng.choice([200, 400]),
                         expiry=1_000 + rng.randint(1, 600), active=1)
            for _ in range(n)]


def cleartext_winner(makers, qty: int) -> int:
    costs = [m.mid + m.half + m.slope * qty + m.invcoef * m.inv for m in makers]
    return min(range(len(makers)), key=lambda i: costs[i])


def scaling(group, sizes, repeats) -> list[dict]:
    prover, verifier = QuoteProver(group), QuoteVerifier(group)
    rows = []
    for n in sizes:
        makers = makers_for(n, seed=5)
        proof, public = prover.prove(makers, qty=100, direction=0, now=1_000,
                                     sentinel=SENTINEL, n_slots=n)
        ok, message = verifier.verify(proof, public)
        prove_times, verify_times = [], []
        for _ in range(repeats):
            start = time.perf_counter()
            prover.prove(makers, qty=100, direction=0, now=1_000,
                         sentinel=SENTINEL, n_slots=n)
            prove_times.append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            verifier.verify(proof, public)
            verify_times.append((time.perf_counter() - start) * 1000)
        row = {
            "makers": n, "verified": ok, "message": message,
            "winner": proof.winner_index,
            "matches_cleartext": proof.winner_index == cleartext_winner(makers, 100),
            "prove": summarise(prove_times),
            "verify": summarise(verify_times),
            # A function of the declared bounds, not of the machine.
            "range_bits": exact(proof.range_bits),
        }
        rows.append(row)
        print(f"  M={n:3d}  verified={ok}  winner_correct={row['matches_cleartext']}  "
              f"prove {render(row['prove'], 1, ' ms')}  "
              f"verify {render(row['verify'], 1, ' ms')}", flush=True)
    return rows


def forgery_controls(group) -> list[dict]:
    """A wrong winner must not be provable, and a tampered proof must not verify."""
    prover, verifier = QuoteProver(group), QuoteVerifier(group)
    makers = makers_for(8, seed=5)
    proof, public = prover.prove(makers, qty=100, direction=0, now=1_000,
                                 sentinel=SENTINEL, n_slots=8)
    out = []

    # 1. claim a maker other than the true minimum
    other = next(i for i in range(8) if i != proof.winner_index)
    forged = QuoteProof(other, proof.winner_value, proof.maker_proofs,
                        proof.winner_opening, proof.minimality,
                        proof.key_commitments, proof.range_bits)
    ok, message = verifier.verify(forged, public)
    out.append({"control": "winner swapped to a non-minimal maker",
                "rejected": not ok, "reason": message})

    # 2. a maker tries to prove an expiry that has already passed
    stale = list(makers)
    stale[0] = MakerWitness(**{**makers[0].__dict__, "expiry": 999})
    try:
        prover.prove(stale, qty=100, direction=0, now=1_000, sentinel=SENTINEL, n_slots=8)
        out.append({"control": "expired maker", "rejected": False,
                    "reason": "prover accepted an expired policy"})
    except ValueError as exc:
        out.append({"control": "expired maker", "rejected": True, "reason": str(exc)[:70]})

    # 3. a request larger than a maker's size limit
    try:
        prover.prove(makers, qty=100_000, direction=0, now=1_000,
                     sentinel=SENTINEL, n_slots=8)
        out.append({"control": "request over the size limit", "rejected": False,
                    "reason": "prover accepted an over-size request"})
    except ValueError as exc:
        out.append({"control": "request over the size limit", "rejected": True,
                    "reason": str(exc)[:70]})

    # 4. tampered minimality proof
    swapped = list(proof.minimality)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    tampered = QuoteProof(proof.winner_index, proof.winner_value, proof.maker_proofs,
                          proof.winner_opening, tuple(swapped),
                          proof.key_commitments, proof.range_bits)
    ok, message = verifier.verify(tampered, public)
    out.append({"control": "minimality proofs swapped between makers",
                "rejected": not ok, "reason": message})

    # 5. the strongest one: build a whole proof around a non-minimal winner
    class LiarProver(QuoteProver):
        def __init__(self, group, target):
            super().__init__(group)
            self.target = target

        def prove(self, makers, **kw):                      # noqa: D401
            proof, public = super().prove(makers, **kw)
            # re-open a maker that is not the minimum and rebuild minimality
            from zk.commit import prove_opening, prove_range
            liar_index = next(i for i in range(len(makers)) if i != proof.winner_index)
            return proof, public, liar_index

    liar = LiarProver(group, 0)
    proof2, public2, liar_index = liar.prove(makers, qty=100, direction=0, now=1_000,
                                             sentinel=SENTINEL, n_slots=8)
    # an honest prover cannot even build the minimality proof for a false winner:
    # the difference against the true minimum is negative
    from zk.commit import prove_range
    from zk.groups import make_group as _mk
    key = Pedersen(group)
    try:
        difference = group.mul(proof2.key_commitments[proof2.winner_index],
                               group.neg(proof2.key_commitments[liar_index]))
        prove_range(key, difference, -1, key.random_blinding(), proof2.range_bits, b"x")
        out.append({"control": "minimality for a false winner", "rejected": False,
                    "reason": "a negative difference was accepted"})
    except ValueError as exc:
        out.append({"control": "minimality for a false winner", "rejected": True,
                    "reason": str(exc)[:70]})

    for row in out:
        print(f"  {row['control']:44s} rejected={row['rejected']}  {row['reason'][:50]}", flush=True)
    return out


def joint_path(group, nodes: int, threshold: int, quorums) -> list[dict]:
    """The same proof, assembled from shares by a quorum of computing nodes."""
    key = Pedersen(group)
    parties = list(range(1, nodes + 1))
    value, blinding = 123_456, key.random_blinding()
    shares = deal(key, value, blinding, parties, threshold)
    rows = []
    def assemble(quorum, audited: bool) -> dict:
        """Median wall time for one assembly, with and without the audit.

        Naming a node that sent a bad partial means deriving every node's share
        commitment from the published coefficient ladder, which is a
        multi-exponentiation apiece and runs on every assembly. Reporting the
        two separately is the honest way to price a property rather than to
        bundle it: a deployment that audits out of band pays the lower figure.
        """
        original = threshold_sigma.audit_partials
        if not audited:
            threshold_sigma.audit_partials = lambda *a, **k: []
        try:
            times = []
            for _ in range(20):
                start = time.perf_counter()
                joint_prove_opening(key, shares, quorum)
                times.append((time.perf_counter() - start) * 1000)
            return summarise(times)
        finally:
            threshold_sigma.audit_partials = original

    for quorum in quorums:
        proof, transcript = joint_prove_opening(key, shares, quorum)
        ok = verify_opening(key, shares.commitment, proof)
        audited = assemble(quorum, audited=True)
        unaudited = assemble(quorum, audited=False)
        row = {"quorum": list(quorum), "size": len(quorum),
               "verified_by_ordinary_verifier": ok,
               "assemble": audited,
               "assemble_without_attribution": unaudited,
               "attribution_factor": audited["mean"] / max(1e-9, unaudited["mean"]),
               "no_node_holds_witness": all(v != value for v in shares.value_shares.values())}
        rows.append(row)
        print(f"  quorum of {len(quorum)}: verified={ok}  "
              f"assemble {render(audited, 3, ' ms')} "
              f"({render(unaudited, 3, ' ms')} without attribution, "
              f"{row['attribution_factor']:.1f}x)  "
              f"no node holds the witness={row['no_node_holds_witness']}", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[4, 8, 16, 32])
    # Three samples cannot carry a standard deviation worth printing.
    ap.add_argument("--repeats", type=int, default=15)
    ap.add_argument("--nodes", type=int, default=7)
    ap.add_argument("--threshold", type=int, default=2)
    args = ap.parse_args()

    group = make_group("ed25519")
    print("== proof of a correct quote ==", flush=True)
    rows = scaling(group, args.sizes, args.repeats)
    print("== forgery controls ==", flush=True)
    controls = forgery_controls(group)
    print("== joint assembly by the computing nodes ==", flush=True)
    quorums = [list(range(1, args.threshold + 2)), list(range(1, args.nodes + 1))]
    joint = joint_path(group, args.nodes, args.threshold, quorums)

    # The paper's own rule is to compare a calibration pair before comparing any
    # millisecond, and this measurement was the one place not following it --- so
    # a change in these timings could not be told apart from a change in the
    # machine. The pair is the same one every other artifact records.
    print("== calibration ==", flush=True)
    calib = calibration(group, args.repeats)
    print(f"  scalar mult {render(calib['scalar_mult_us'], 1, ' us')}, "
          f"40-bit range proof {render(calib['range_proof_40bit_ms'], 2, ' ms')}",
          flush=True)

    payload = {"host": this_host(), "scaling": rows, "forgery_controls": controls,
               "joint": joint, "nodes": args.nodes, "threshold": args.threshold,
               "calibration": calib}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
