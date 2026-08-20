"""zkPI: an instruction a settlement venue can check without reading it.

The point of these tests is the pair of properties that make the construction
pluggable: a venue needs only `verify` and the nullifier, and everything it
could learn from the instruction is either a commitment or a one-time tag.
"""

from __future__ import annotations

import secrets
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from zk.commit import Pedersen                                              # noqa: E402
from zk.groups import make_group                                            # noqa: E402
from zk.zkpi import (                                                       # noqa: E402
    InstructionBounds, InstructionIssuer, SettlementVenue, ZkPaymentInstruction,
)


@pytest.fixture(scope="module")
def group():
    return make_group("ed25519")


@pytest.fixture()
def parts(group):
    key = Pedersen(group, b"qomm:zkpi:v1")
    # One quorum, dealt once and trusted by the venue. An issuer that deals a
    # fresh one per instruction is an issuer that can sign for itself, and the
    # venue below is constructed to notice.
    secret, blinding = key.random_blinding(), key.random_blinding()
    issuer = InstructionIssuer(group, key, quorum_secret=secret,
                               quorum_blinding=blinding)
    return key, issuer, SettlementVenue(group, key, quorum_key=issuer.quorum_key)


def _issue(issuer, group, **overrides):
    args = dict(asset=3, amount=100, price=99_990,
                payer_handle=group.hash_to_point(b"entity-A"),
                payee_handle=group.hash_to_point(b"entity-B"),
                deadline=1_500, nonce=secrets.token_bytes(32), quote_key=1_599_845,
                nodes=list(range(1, 8)), threshold=2, quorum=[1, 2, 3])
    args.update(overrides)
    return issuer.issue(**args)


def test_a_well_formed_instruction_settles_once(group, parts):
    _, issuer, venue = parts
    instruction, _ = _issue(issuer, group)
    assert venue.verify(instruction, now=1_000) == (True, "ok")
    assert venue.settle(instruction, now=1_000)[0]
    replayed, reason = venue.settle(instruction, now=1_000)
    assert not replayed and "already settled" in reason
    assert venue.spent() == 1


@pytest.mark.parametrize("field,value", [
    ("deadline", 1_400),
    ("nonce", b"\x11" * 32),
])
def test_tampering_breaks_the_quorum_signature(group, parts, field, value):
    _, issuer, venue = parts
    instruction, _ = _issue(issuer, group)
    broken = ZkPaymentInstruction(**{**instruction.__dict__, field: value})
    ok, reason = venue.verify(broken, now=1_000)
    assert not ok and "does not cover this instruction" in reason


def test_a_signature_cannot_be_moved_to_another_instruction(group, parts):
    _, issuer, venue = parts
    first, _ = _issue(issuer, group)
    second, _ = _issue(issuer, group, amount=200)
    forged = ZkPaymentInstruction(
        **{**second.__dict__, "quorum_signature": first.quorum_signature})
    ok, reason = venue.verify(forged, now=1_000)
    assert not ok and "does not cover this instruction" in reason


def test_an_amount_outside_the_published_bounds_cannot_be_issued(group, parts):
    _, issuer, _ = parts
    with pytest.raises(ValueError):
        _issue(issuer, group, amount=99_999_999)


def test_an_expired_or_far_future_deadline_is_refused(group, parts):
    _, issuer, venue = parts
    instruction, _ = _issue(issuer, group)
    assert not venue.verify(instruction, now=1_600)[0]          # already past
    assert venue.verify(instruction, now=0)[0]                  # inside the horizon
    distant, _ = _issue(issuer, group, deadline=100_000)
    ok, reason = venue.verify(distant, now=1_000)
    assert not ok and "horizon" in reason                       # beyond the horizon


def test_payer_and_payee_must_differ(group, parts):
    _, issuer, venue = parts
    handle = group.hash_to_point(b"entity-A")
    instruction, _ = _issue(issuer, group, payer_handle=handle, payee_handle=handle)
    ok, reason = venue.verify(instruction, now=1_000)
    assert not ok and "same entity" in reason


def test_a_short_nonce_is_refused(group, parts):
    _, issuer, venue = parts
    instruction, _ = _issue(issuer, group, nonce=b"\x01" * 8)
    ok, reason = venue.verify(instruction, now=1_000)
    assert not ok and "one-time" in reason


def test_two_instructions_from_one_payer_have_different_nullifiers(group, parts):
    _, issuer, venue = parts
    first, _ = _issue(issuer, group)
    second, _ = _issue(issuer, group)
    assert first.nullifier(group) != second.nullifier(group)
    assert venue.settle(first, now=1_000)[0]
    assert venue.settle(second, now=1_000)[0]


def test_the_instruction_carries_no_plaintext_leg(group, parts):
    """Everything a venue sees is a commitment, a handle or a one-time tag."""
    _, issuer, _ = parts
    instruction, _ = _issue(issuer, group, asset=3, amount=100, price=99_990)
    encoded = b"".join([
        group.encode(instruction.asset_commitment),
        group.encode(instruction.amount_commitment),
        group.encode(instruction.price_commitment),
        group.encode(instruction.payer_handle),
        group.encode(instruction.payee_handle),
    ])
    for secret in (3, 100, 99_990):
        assert secret.to_bytes(4, "big") not in encoded
        assert secret.to_bytes(8, "big") not in encoded


def test_the_same_leg_twice_looks_different(group, parts):
    """Fresh blinding, so identical trades do not produce identical commitments."""
    _, issuer, _ = parts
    first, _ = _issue(issuer, group)
    second, _ = _issue(issuer, group)
    assert group.encode(first.amount_commitment) != group.encode(second.amount_commitment)
    assert group.encode(first.price_commitment) != group.encode(second.price_commitment)


def test_a_venue_needs_only_verify_and_the_nullifier(group, parts):
    """The pluggable interface: no knowledge of how the price was reached.

    It does need to know *whose* quorum it is settling for, though, which is
    the one thing a venue cannot learn from the instruction in front of it.
    """
    _, issuer, _ = parts
    other_venue = SettlementVenue(group, Pedersen(group, b"qomm:zkpi:v1"),
                                  InstructionBounds(),
                                  quorum_key=issuer.quorum_key)
    instruction, _ = _issue(issuer, group)
    assert other_venue.verify(instruction, now=1_000)[0]
    assert isinstance(instruction.nullifier(group), bytes)


def test_a_venue_that_trusts_no_quorum_settles_nothing(group, parts):
    _, issuer, _ = parts
    blind = SettlementVenue(group, Pedersen(group, b"qomm:zkpi:v1"), InstructionBounds())
    instruction, _ = _issue(issuer, group)
    ok, reason = blind.verify(instruction, now=1_000)
    assert not ok and "trusts no quorum" in reason


def test_an_issuer_cannot_deal_itself_a_quorum(group, parts):
    """The forgery the venue used to accept: a quorum invented on the spot.

    Nothing here is malformed. The signature is a real joint opening over a
    real deal; it is simply a deal the issuer made for itself, and the venue
    used to read the key out of the instruction and check against that.
    """
    key, issuer, venue = parts
    impostor = InstructionIssuer(group, key,
                                 quorum_secret=key.random_blinding(),
                                 quorum_blinding=key.random_blinding())
    instruction, _ = _issue(impostor, group)
    ok, reason = venue.verify(instruction, now=1_000)
    assert not ok and "does not trust" in reason


def test_an_issuer_that_deals_a_fresh_quorum_each_time_is_not_trusted(group, parts):
    key, _, venue = parts
    ad_hoc = InstructionIssuer(group, key)          # no fixed quorum: the old behaviour
    instruction, _ = _issue(ad_hoc, group)
    ok, reason = venue.verify(instruction, now=1_000)
    assert not ok and "does not trust" in reason


def test_a_venue_with_tighter_bounds_refuses(group, parts):
    _, issuer, _ = parts
    instruction, _ = _issue(issuer, group, amount=100)
    strict = SettlementVenue(group, Pedersen(group, b"qomm:zkpi:v1"),
                             InstructionBounds(amount=(1, 10)))
    ok, reason = strict.verify(instruction, now=1_000)
    assert not ok and "amount" in reason
