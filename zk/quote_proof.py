"""Publicly verifiable proof that the opened quote is the correct one.

The receipts in `qomm_audit` bind a node to a result; they do not show the result
is right. This module closes that gap for the quote circuit, without a
general-purpose SNARK and without a trusted setup, by proving the statement the
circuit actually computes:

    for each maker i, key_i is the committed policy applied to the committed
    request, and the opened winner is the smallest of those keys.

Every step is a sigma protocol over Pedersen commitments, which matters for two
reasons. Sigma responses are linear in the witness, so the computing nodes can
assemble the proof jointly from shares (see `threshold_sigma`). And the whole
thing is checked by an ordinary verifier with no setup.

Structure per maker:

    depth_i    = slope_i * qty            product proof
    skew_i     = invcoef_i * inv_i        product proof
    ask_i      = mid_i + half_i + depth_i + skew_i        linear, free
    bid_i      = mid_i - half_i - depth_i + skew_i        linear, free
    fits_i     = maxqty_i - qty >= 0      range proof
    fresh_i    = expiry_i - now  >= 0     range proof
    ok_i       is a bit, and gates the cost                bit + product proofs
    key_i      = cost_i * M + i           linear, free

and then over the whole set:

    winner     key_j opens to the revealed value            opening proof
    minimality key_i - v >= 0 for every i                   range proofs

Minimality plus membership is exactly "v is the minimum", so an incorrect winner
cannot be proved.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .commit import (
    BitProof, OpeningProof, Pedersen, ProductProof, RangeProof, prove_bit,
    prove_opening, prove_product, prove_range, shift_commitment, verify_bit,
    verify_opening, verify_product, verify_range,
)
from .groups import Group


FIELDS = ("mid", "half", "slope", "invcoef", "inv", "maxqty", "expiry", "active")


@dataclass(frozen=True)
class MakerWitness:
    """A maker's policy and the blindings it registered it under.

    The blindings are what makes this a witness for a *registered* policy
    rather than for whatever the prover felt like committing. Without them the
    prover drew fresh blindings for every field at proving time, so the quote
    proof established a minimum over a set of commitments the prover had just
    invented -- true about those, and about nothing on the record.
    """

    mid: int
    half: int
    slope: int
    invcoef: int
    inv: int
    maxqty: int
    expiry: int
    active: int
    blindings: dict = field(default_factory=dict)

    def registered(self, key: Pedersen) -> dict:
        """The commitments this policy was registered under."""
        return {name: key.commit(getattr(self, name), self.blindings[name])
                for name in FIELDS}


def registry_digest(group: Group, registered: Sequence[Mapping[str, Any]]) -> bytes:
    """One digest over the whole eligible set, in order.

    Fixing this in the public statement is what makes maker *omission* visible:
    a prover that drops a maker to change the winner has to publish a different
    digest, and the digest was agreed before the request arrived.
    """
    hasher = hashlib.sha256(b"QOMM:QUOTE:REGISTRY:v1")
    hasher.update(len(registered).to_bytes(4, "big"))
    for policy in registered:
        for name in FIELDS:
            hasher.update(group.encode(policy[name]))
    return hasher.digest()


@dataclass(frozen=True)
class MakerProof:
    depth: ProductProof
    skew: ProductProof
    gate_cost: ProductProof
    fits: RangeProof
    fresh: RangeProof
    active_bit: BitProof
    ok_bit: BitProof
    commitments: dict


@dataclass(frozen=True)
class QuoteProof:
    winner_index: int
    winner_value: int
    maker_proofs: tuple
    winner_opening: OpeningProof
    minimality: tuple            # one range proof per maker
    key_commitments: tuple
    range_bits: int


class QuoteProver:
    """Builds the proof. The witness may be held by one party or shared."""

    def __init__(self, group: Group, key: Pedersen | None = None,
                 sentinel_bits: int = 24):
        self.group = group
        self.key = key or Pedersen(group, b"qomm:policy:v1")
        self.sentinel_bits = sentinel_bits

    def _blind(self) -> int:
        return self.key.random_blinding()

    def prove(self, makers: Sequence[MakerWitness], *, qty: int, direction: int,
              now: int, sentinel: int, n_slots: int,
              market_digest: bytes = b"", slot: int = 0,
              context: bytes = b"") -> tuple[QuoteProof, dict]:
        key = self.key
        group = self.group
        order = group.order

        for index, maker in enumerate(makers):
            missing = [name for name in FIELDS if name not in maker.blindings]
            if missing:
                raise ValueError(
                    f"maker {index} has no registered blinding for {missing}: a "
                    "quote proof is about policies that were put on the record, "
                    "and a witness without them is a policy invented now")

        r_qty = self._blind()
        c_qty = key.commit(qty, r_qty)

        keys: list[int] = []
        key_blindings: list[int] = []
        key_commitments: list[Any] = []
        maker_proofs: list[MakerProof] = []

        for index, maker in enumerate(makers):
            tag = context + b":mm:" + index.to_bytes(2, "big")
            # Every field opens the commitment the maker registered. Drawing
            # fresh blindings here is what made the proof true about a set the
            # prover had just made up.
            blind = maker.blindings
            r_slope, r_invcoef, r_inv = (blind["slope"], blind["invcoef"], blind["inv"])
            c_slope = key.commit(maker.slope, r_slope)
            c_invcoef = key.commit(maker.invcoef, r_invcoef)
            c_inv = key.commit(maker.inv, r_inv)

            r_depth = self._blind()
            depth = maker.slope * qty
            depth_proof = prove_product(key, c_slope, maker.slope, r_slope,
                                        qty, r_qty, r_depth, tag + b":depth")
            r_skew = self._blind()
            skew = maker.invcoef * maker.inv
            skew_proof = prove_product(key, c_invcoef, maker.invcoef, r_invcoef,
                                       maker.inv, r_inv, r_skew, tag + b":skew")

            r_mid, r_half = blind["mid"], blind["half"]
            ask = maker.mid + maker.half + depth + skew
            bid = maker.mid - maker.half - depth + skew
            r_ask = (r_mid + r_half + r_depth + r_skew) % order
            r_bid = (r_mid - r_half - r_depth + r_skew) % order

            # eligibility, each piece proved rather than asserted
            r_maxqty, r_expiry = blind["maxqty"], blind["expiry"]
            c_fits = key.commit(maker.maxqty - qty, (r_maxqty - r_qty) % order)
            fits_proof = prove_range(key, c_fits, maker.maxqty - qty,
                                     (r_maxqty - r_qty) % order, self.sentinel_bits,
                                     tag + b":fits")
            c_fresh = key.commit(maker.expiry - now, r_expiry)
            fresh_proof = prove_range(key, c_fresh, maker.expiry - now, r_expiry,
                                      self.sentinel_bits, tag + b":fresh")

            r_active = blind["active"]
            c_active = key.commit(maker.active, r_active)
            active_proof = prove_bit(key, c_active, maker.active, r_active, tag + b":active")

            ok = 1 if (maker.active == 1 and qty <= maker.maxqty and maker.expiry > now) else 0
            r_ok = self._blind()
            c_ok = key.commit(ok, r_ok)
            ok_proof = prove_bit(key, c_ok, ok, r_ok, tag + b":ok")

            cost = -bid if direction == 1 else ask
            r_cost = (-r_bid if direction == 1 else r_ask) % order
            c_cost = key.commit(cost, r_cost)

            # gated = ok * (cost - sentinel), so gated + sentinel is the effective cost
            r_gated = self._blind()
            gated_proof = prove_product(key, c_ok, ok, r_ok, cost - sentinel,
                                        (r_cost - 0) % order, r_gated, tag + b":gate")
            effective = ok * (cost - sentinel) + sentinel
            r_effective = r_gated

            packed = effective * n_slots + index
            r_packed = (r_effective * n_slots) % order
            keys.append(packed)
            key_blindings.append(r_packed)
            key_commitments.append(key.commit(packed, r_packed))

            maker_proofs.append(MakerProof(
                depth=depth_proof, skew=skew_proof, gate_cost=gated_proof,
                fits=fits_proof, fresh=fresh_proof,
                active_bit=active_proof, ok_bit=ok_proof,
                commitments={
                    "mid": key.commit(maker.mid, r_mid),
                    "half": key.commit(maker.half, r_half),
                    "maxqty": key.commit(maker.maxqty, r_maxqty),
                    "expiry": key.commit(maker.expiry, r_expiry),
                    "slope": c_slope, "invcoef": c_invcoef, "inv": c_inv,
                    "depth": key.commit(depth, r_depth), "skew": key.commit(skew, r_skew),
                    "fits": c_fits, "fresh": c_fresh, "active": c_active, "ok": c_ok,
                    "cost": c_cost, "gated": key.commit(ok * (cost - sentinel), r_gated),
                    "shifted_cost": key.commit(cost - sentinel, r_cost),
                }))

        winner = min(range(len(keys)), key=lambda i: keys[i])
        value = keys[winner]
        # Bind the *published* number, not merely the commitment. An opening
        # proof shows knowledge of some opening and says nothing about which, so
        # proving C_winner directly would leave the price a free parameter: a
        # venue could publish any figure and the proof would still verify.
        # Proving that C_winner / g^value is a pure power of h says the
        # commitment opens to this value and no other, at the same cost.
        winner_opening = prove_opening(
            key, shift_commitment(key, key_commitments[winner], value),
            0, key_blindings[winner], context + b":winner")

        span_bits = max(1, (sentinel * n_slots * 2).bit_length())
        minimality = []
        for index in range(len(keys)):
            difference = keys[index] - value
            r_diff = (key_blindings[index] - key_blindings[winner]) % order
            c_diff = shift_commitment(key, key_commitments[index], 0)
            c_diff = self.group.mul(c_diff, self.group.neg(key_commitments[winner]))
            minimality.append(prove_range(key, c_diff, difference, r_diff, span_bits,
                                          context + b":min:" + index.to_bytes(2, "big")))

        proof = QuoteProof(winner, value, tuple(maker_proofs), winner_opening,
                           tuple(minimality), tuple(key_commitments), span_bits)
        registered = [maker.registered(key) for maker in makers]
        public = {"qty_commitment": c_qty, "now": now, "sentinel": sentinel,
                  "n_slots": n_slots, "direction": direction,
                  # what the proof is *about*, as opposed to what it proves
                  "registry": registered,
                  "registry_digest": registry_digest(group, registered),
                  "market_digest": market_digest,
                  "slot": slot}
        return proof, public


class QuoteVerifier:
    def __init__(self, group: Group, key: Pedersen | None = None,
                 sentinel_bits: int = 24):
        self.group = group
        self.key = key or Pedersen(group, b"qomm:policy:v1")
        self.sentinel_bits = sentinel_bits

    def verify(self, proof: QuoteProof, public: Mapping[str, Any],
               context: bytes = b"") -> tuple[bool, str]:
        key = self.key
        group = self.group
        c_qty = public["qty_commitment"]

        # What the statement has to say before any of it means anything: which
        # policies, which set, and which request. Without these the proof
        # established a minimum over commitments the prover chose, which is
        # true and says nothing about the market that was registered.
        registered = public.get("registry")
        if registered is None:
            return False, ("the statement names no registered policies, so this "
                           "proof is about commitments the prover chose")
        if len(registered) != len(proof.maker_proofs):
            return False, (f"the statement registers {len(registered)} makers and "
                           f"the proof covers {len(proof.maker_proofs)}")
        expected = registry_digest(group, registered)
        if public.get("registry_digest") != expected:
            return False, "the registered set is not the one this statement names"

        for index, maker in enumerate(proof.maker_proofs):
            tag = context + b":mm:" + index.to_bytes(2, "big")
            c = maker.commitments
            for name in FIELDS:
                if name not in c:
                    return False, f"maker {index}: the proof does not carry {name}"
                if group.encode(c[name]) != group.encode(registered[index][name]):
                    return False, (f"maker {index}: {name} is not the one on the "
                                   "register")
            if not verify_product(key, c["slope"], c_qty, c["depth"], maker.depth,
                                  tag + b":depth"):
                return False, f"maker {index}: depth is not slope * quantity"
            if not verify_product(key, c["invcoef"], c["inv"], c["skew"], maker.skew,
                                  tag + b":skew"):
                return False, f"maker {index}: skew is not invcoef * inventory"
            if not verify_range(key, c["fits"], maker.fits, tag + b":fits"):
                return False, f"maker {index}: size limit not shown to hold"
            if not verify_range(key, c["fresh"], maker.fresh, tag + b":fresh"):
                return False, f"maker {index}: expiry not shown to be in the future"
            if not verify_bit(key, c["active"], maker.active_bit, tag + b":active"):
                return False, f"maker {index}: active flag is not a bit"
            if not verify_bit(key, c["ok"], maker.ok_bit, tag + b":ok"):
                return False, f"maker {index}: eligibility flag is not a bit"
            if not verify_product(key, c["ok"], c["shifted_cost"], c["gated"],
                                  maker.gate_cost, tag + b":gate"):
                return False, f"maker {index}: cost is not gated by eligibility"

        # The verifier reconstructs the same residual from the published value,
        # so a proof made for one price does not carry to another.
        if not verify_opening(key, shift_commitment(
                                  key, proof.key_commitments[proof.winner_index],
                                  proof.winner_value),
                              proof.winner_opening, context + b":winner"):
            return False, "the published winner value is not what the commitment opens to"

        for index, range_proof in enumerate(proof.minimality):
            difference = group.mul(proof.key_commitments[index],
                                   group.neg(proof.key_commitments[proof.winner_index]))
            if not verify_range(key, difference, range_proof,
                                context + b":min:" + index.to_bytes(2, "big")):
                return False, f"maker {index}: not shown to be at least the winner"
        return True, "ok"
