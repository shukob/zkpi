"""The proof that the opened quote is correct, and that a wrong one is not provable."""

from __future__ import annotations

import sys
from pathlib import Path

import copy
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from zk.commit import Pedersen, prove_product, verify_opening, verify_product  # noqa: E402
from zk.groups import make_group                                               # noqa: E402
from zk.quote_proof import MakerWitness, QuoteProof, QuoteProver, QuoteVerifier  # noqa: E402
from zk.quote_proof import FIELDS, registry_digest  # noqa: E402
from zk.threshold_sigma import deal, joint_prove_opening, lagrange_at_zero      # noqa: E402

SENTINEL = 1 << 20


@pytest.fixture(scope="module")
def group():
    return make_group("ed25519")


def _register(group, values: dict) -> MakerWitness:
    """A policy as it goes on the record: values and the blindings that hide them."""
    key = Pedersen(group, b"qomm:quote:v1")
    return MakerWitness(**values,
                        blindings={name: key.random_blinding() for name in FIELDS})


def _makers(n: int = 6, group=None):
    group = group or make_group("ed25519")
    return [_register(group, dict(mid=100_000 + 3 * i, half=10 + i, slope=i % 3,
                                  invcoef=i % 2, inv=10 * i, maxqty=400,
                                  expiry=1_600, active=1))
            for i in range(n)]


def _prove(group, makers=None, **kw):
    makers = makers or _makers()
    options = dict(qty=100, direction=0, now=1_000, sentinel=SENTINEL, n_slots=len(makers))
    options.update(kw)
    return QuoteProver(group).prove(makers, **options)


def test_an_honest_quote_verifies(group):
    proof, public = _prove(group)
    ok, reason = QuoteVerifier(group).verify(proof, public)
    assert ok, reason


def test_the_proved_winner_is_the_cleartext_minimum(group):
    makers = _makers(8)
    proof, _ = _prove(group, makers)
    costs = [m.mid + m.half + m.slope * 100 + m.invcoef * m.inv for m in makers]
    assert proof.winner_index == min(range(len(makers)), key=lambda i: costs[i])


def test_a_sell_uses_the_other_side(group):
    makers = _makers(6)
    buy, _ = _prove(group, makers, direction=0)
    sell, _ = _prove(group, makers, direction=1)
    bids = [m.mid - m.half - m.slope * 100 + m.invcoef * m.inv for m in makers]
    assert sell.winner_index == max(range(len(makers)), key=lambda i: bids[i])
    assert buy.winner_value != sell.winner_value


def test_claiming_a_different_winner_fails(group):
    proof, public = _prove(group)
    other = next(i for i in range(len(proof.key_commitments)) if i != proof.winner_index)
    forged = QuoteProof(other, proof.winner_value, proof.maker_proofs,
                        proof.winner_opening, proof.minimality,
                        proof.key_commitments, proof.range_bits)
    ok, reason = QuoteVerifier(group).verify(forged, public)
    assert not ok and "winner" in reason


def test_swapping_minimality_proofs_fails(group):
    proof, public = _prove(group)
    swapped = list(proof.minimality)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    tampered = QuoteProof(proof.winner_index, proof.winner_value, proof.maker_proofs,
                          proof.winner_opening, tuple(swapped),
                          proof.key_commitments, proof.range_bits)
    ok, reason = QuoteVerifier(group).verify(tampered, public)
    assert not ok and "at least the winner" in reason


def test_an_ineligible_maker_is_provable_and_cannot_win(group):
    """It used to be unprovable, which left leaving it out as the only option.

    A negative difference has no range proof, so an expired or oversized maker
    could not appear in a quote at all --- and a quote that silently drops a
    maker is exactly what the register exists to catch. Eligibility is now a
    bit that can be zero, so the maker appears, is gated to the sentinel, and
    loses.
    """
    makers = _makers(group=group)
    makers[0] = MakerWitness(**{**makers[0].__dict__, "expiry": 999})
    proof, public = _prove(group, makers, now=1_000)
    ok, why = QuoteVerifier(group).verify(proof, public)
    assert ok, why
    assert proof.winner_index != 0, "an expired maker won the quote"


def test_a_maker_over_the_size_limit_is_provable_and_cannot_win(group):
    makers = _makers(group=group)
    makers[0] = MakerWitness(**{**makers[0].__dict__, "maxqty": 10})
    proof, public = _prove(group, makers, qty=100)
    ok, why = QuoteVerifier(group).verify(proof, public)
    assert ok, why
    assert proof.winner_index != 0, "a maker over its size limit won the quote"


def test_a_request_beyond_every_maker_gets_no_quote(group):
    """Every maker gates off, so every key is the sentinel and nobody wins on
    price. The circuit answers "no quote" rather than refusing to answer."""
    proof, public = _prove(group, qty=100_000)
    ok, why = QuoteVerifier(group).verify(proof, public)
    assert ok, why
    assert proof.winner_value >= public["sentinel"], (
        "a request nobody can fill produced a real price")


def test_a_tampered_product_proof_fails(group):
    proof, public = _prove(group)
    victim = proof.maker_proofs[0]
    broken = type(victim)(**{**victim.__dict__,
                             "depth": proof.maker_proofs[1].depth})
    tampered = QuoteProof(proof.winner_index, proof.winner_value,
                          (broken,) + proof.maker_proofs[1:], proof.winner_opening,
                          proof.minimality, proof.key_commitments, proof.range_bits)
    ok, reason = QuoteVerifier(group).verify(tampered, public)
    assert not ok and "depth" in reason


def test_product_proof_rejects_a_wrong_product(group):
    key = Pedersen(group)
    a, b = 7, 11
    r_a, r_b, r_c = (key.random_blinding() for _ in range(3))
    c_a, c_b = key.commit(a, r_a), key.commit(b, r_b)
    proof = prove_product(key, c_a, a, r_a, b, r_b, r_c)
    assert verify_product(key, c_a, c_b, key.commit(a * b, r_c), proof)
    assert not verify_product(key, c_a, c_b, key.commit(a * b + 1, r_c), proof)


# --- joint assembly by the computing nodes --------------------------------

def test_a_quorum_assembles_a_proof_the_ordinary_verifier_accepts(group):
    key = Pedersen(group)
    parties = list(range(1, 8))
    value, blinding = 4_242, key.random_blinding()
    shares = deal(key, value, blinding, parties, threshold=2)
    for quorum in ([1, 2, 3], [2, 5, 7], [4, 5, 6, 7], parties):
        proof, transcript = joint_prove_opening(key, shares, quorum)
        assert verify_opening(key, shares.commitment, proof)
        assert transcript["quorum"] == list(quorum)


def test_no_single_node_holds_the_witness(group):
    key = Pedersen(group)
    parties = list(range(1, 8))
    value = 4_242
    shares = deal(key, value, key.random_blinding(), parties, threshold=2)
    assert all(share != value for share in shares.value_shares.values())
    # fewer than threshold+1 shares determine nothing
    assert len(set(shares.value_shares.values())) == len(parties)


def test_below_threshold_cannot_assemble(group):
    key = Pedersen(group)
    parties = list(range(1, 8))
    shares = deal(key, 99, key.random_blinding(), parties, threshold=2)
    proof, _ = joint_prove_opening(key, shares, [1, 2])      # one short of threshold+1
    assert not verify_opening(key, shares.commitment, proof)


def test_lagrange_reconstructs_the_constant_term(group):
    order = group.order
    coefficients = lagrange_at_zero([1, 2, 3], order)
    # f(x) = 5 + 7x + 11x^2 evaluated at 1,2,3 must interpolate back to 5
    values = {x: (5 + 7 * x + 11 * x * x) % order for x in (1, 2, 3)}
    assert sum(coefficients[x] * values[x] for x in values) % order == 5


def test_the_published_price_is_bound_by_the_proof():
    """An opening proof shows knowledge of *some* opening and says nothing about
    which. Binding only the commitment therefore leaves the published price a
    free parameter, and a venue could quote whatever it liked while the proof
    still verified. The residual C/g^value is what closes it."""
    group = make_group("ed25519")
    prover, verifier = QuoteProver(group), QuoteVerifier(group)
    makers = [_register(group, dict(mid=0, half=h, slope=1, invcoef=1, inv=10,
                                    maxqty=1_000, expiry=10_000, active=1))
              for h in (8, 5, 12)]
    proof, public = prover.prove(makers, qty=100, direction=0, now=1_000,
                                 sentinel=1 << 20, n_slots=4)
    assert verifier.verify(proof, public)[0]

    for delta in (1, -1, 999_999):
        lied = QuoteProof(proof.winner_index, proof.winner_value + delta,
                          proof.maker_proofs, proof.winner_opening, proof.minimality,
                          proof.key_commitments, proof.range_bits)
        ok, reason = verifier.verify(lied, public)
        assert not ok, f"a price off by {delta} was accepted"
        assert "winner value" in reason


# --- the proof has to be about the policies that were registered ------------
#
# The statement used to carry the quantity commitment, the clock, a sentinel, a
# slot count and a direction. Every commitment the minimum was taken over was
# drawn by the prover at proving time, so the proof said "among these numbers I
# just committed to, this is the smallest" -- true, and silent about whether
# those numbers were the market's.

def test_a_statement_with_no_registry_is_refused(group):
    proof, public = _prove(group)
    stripped = {k: v for k, v in public.items() if k != "registry"}
    ok, why = QuoteVerifier(group).verify(proof, stripped)
    assert not ok and "no registered policies" in why


def test_a_policy_the_register_does_not_hold_is_refused(group):
    """The forgery: quote from a policy that was never registered."""
    makers = _makers(group=group)
    proof, public = _prove(group, makers)
    key = Pedersen(group, b"qomm:quote:v1")
    swapped = [dict(policy) for policy in public["registry"]]
    swapped[0]["half"] = key.commit(1, key.random_blinding())
    moved = dict(public, registry=swapped,
                 registry_digest=registry_digest(group, swapped))
    ok, why = QuoteVerifier(group).verify(proof, moved)
    assert not ok and "not the one on the register" in why


def test_a_registry_digest_that_does_not_cover_the_registry_is_refused(group):
    proof, public = _prove(group)
    ok, why = QuoteVerifier(group).verify(proof, dict(public, registry_digest=b"\x00" * 32))
    assert not ok and "not the one this statement names" in why


def test_dropping_a_maker_from_the_set_is_visible(group):
    """Omission changes the digest, and the digest was fixed before the request."""
    makers = _makers(group=group)
    proof, public = _prove(group, makers)
    fewer = public["registry"][:-1]
    ok, why = QuoteVerifier(group).verify(
        proof, dict(public, registry=fewer,
                    registry_digest=registry_digest(group, fewer)))
    assert not ok and "registers" in why


def test_a_witness_without_registered_blindings_cannot_prove(group):
    """A policy invented at proving time is not a registered one."""
    bare = [MakerWitness(mid=0, half=5, slope=1, invcoef=1, inv=10,
                         maxqty=1_000, expiry=10_000, active=1)]
    with pytest.raises(ValueError, match="put on the record"):
        QuoteProver(group).prove(bare, qty=10, direction=0, now=1_000,
                                 sentinel=1 << 20, n_slots=2)


def test_the_registry_digest_is_a_function_of_the_whole_set_and_its_order(group):
    a = [m.registered(Pedersen(group, b"qomm:quote:v1")) for m in _makers(3, group)]
    assert registry_digest(group, a) == registry_digest(group, a)
    assert registry_digest(group, a) != registry_digest(group, list(reversed(a)))
    assert registry_digest(group, a) != registry_digest(group, a[:-1])


# --- an eligible maker cannot be switched off -------------------------------
#
# `ok` used to be committed with a bit proof and nothing else. `fits` and
# `fresh` were shown non-negative and never tied to it, so setting `ok = 0` for
# an eligible maker left both range proofs passing on non-negative values and
# gated that maker to the sentinel. Suppressing the best price was free.

def test_the_eligibility_bit_is_the_conjunction_and_not_a_choice(group):
    """The forgery: gate off the maker that would have won."""
    makers = _makers(group=group)
    proof, public = _prove(group, makers)
    verifier = QuoteVerifier(group)
    assert verifier.verify(proof, public)[0]

    winner = proof.winner_index
    key = Pedersen(group, b"qomm:quote:v1")
    forged = copy.deepcopy(proof)
    target = forged.maker_proofs[winner]
    blinding = key.random_blinding()
    target.commitments["ok"] = key.commit(0, blinding)
    ok, why = verifier.verify(forged, public)
    assert not ok, "the winning maker was gated off and the proof still verified"
    assert "conjunction" in why or "gated" in why, why


def test_the_size_test_is_taken_from_the_register_not_from_the_proof(group):
    """A prover that picks what it proves eligibility about picks the answer."""
    makers = _makers(group=group)
    proof, public = _prove(group, makers)
    key = Pedersen(group, b"qomm:quote:v1")
    forged = copy.deepcopy(proof)
    forged.maker_proofs[0].commitments["fits"] = key.commit(10 ** 6,
                                                            key.random_blinding())
    ok, why = QuoteVerifier(group).verify(forged, public)
    assert not ok and "maxqty - quantity" in why


def test_the_freshness_test_is_taken_from_the_register_and_the_clock(group):
    makers = _makers(group=group)
    proof, public = _prove(group, makers)
    key = Pedersen(group, b"qomm:quote:v1")
    forged = copy.deepcopy(proof)
    forged.maker_proofs[0].commitments["fresh_t_input"] = key.commit(
        10 ** 6, key.random_blinding())
    ok, why = QuoteVerifier(group).verify(forged, public)
    assert not ok and "expiry - now - 1" in why
