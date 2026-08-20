"""A bad partial must be attributable by someone who holds no secret.

The joint proof exists so that no node holds the witness. An attribution check
that needs the witness is therefore a check nobody can run: not the venue, not a
verifier, not the honest nodes. It used to take the shares, so a quorum whose
proof failed could report only that somebody had cheated.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zk.commit import Pedersen, verify_opening                     # noqa: E402
from zk.groups import make_group                                   # noqa: E402
from zk.threshold_sigma import (                                   # noqa: E402
    audit_partials, deal, joint_prove_opening, share_commitment, verify_share,
)

PARTIES = [1, 2, 3, 4, 5]
QUORUM = [1, 2, 3]


@pytest.fixture(scope="module")
def key():
    return Pedersen(make_group("ed25519"), b"qomm:attribution:test")


def _shares(key, value=1234):
    return deal(key, value, key.random_blinding(), PARTIES, threshold=2)


def test_share_commitments_are_public(key):
    """Derived from the published ladder, they match what the shares open to."""
    shares = _shares(key)
    for party in PARTIES:
        derived = share_commitment(key, shares.coefficient_commitments, party)
        actual = key.commit(shares.value_shares[party], shares.blinding_shares[party])
        assert key.group.encode(derived) == key.group.encode(actual)
        assert verify_share(key, shares, party)


def test_the_ladder_starts_at_the_public_commitment(key):
    """Evaluating at zero is the secret's own commitment, so nothing new leaks."""
    shares = _shares(key)
    assert key.group.encode(shares.coefficient_commitments[0]) == \
        key.group.encode(shares.commitment)


def test_an_honest_quorum_names_nobody(key):
    shares = _shares(key)
    proof, transcript = joint_prove_opening(key, shares, QUORUM, context=b"ctx")
    assert transcript["bad_partials"] == []
    assert verify_opening(key, shares.commitment, proof, b"ctx")


def test_a_bad_partial_is_named_and_the_proof_fails(key):
    shares = _shares(key)
    _, honest = joint_prove_opening(key, shares, QUORUM, context=b"ctx")
    good = tuple(honest["partial_responses"][2])
    proof, transcript = joint_prove_opening(
        key, shares, QUORUM, context=b"ctx",
        faulty={2: (good[0] + 1, good[1])})
    assert transcript["bad_partials"] == [2]
    assert not verify_opening(key, shares.commitment, proof, b"ctx")


def test_an_observer_with_no_secrets_can_attribute(key):
    """The point of the change: only the transcript and the ladder are needed."""
    shares = _shares(key)
    _, honest = joint_prove_opening(key, shares, QUORUM, context=b"ctx")
    good = tuple(honest["partial_responses"][3])
    _, transcript = joint_prove_opening(
        key, shares, QUORUM, context=b"ctx",
        faulty={3: (good[0], good[1] + 7)})

    # everything below is public: the ladder, the partial commitments, the
    # responses and the challenge. No share is in scope.
    group = key.group
    partial_commitments = {
        p: group.decode(bytes.fromhex(v))
        for p, v in transcript["partial_commitments"].items()}
    named = audit_partials(
        key, shares.coefficient_commitments, transcript["quorum"],
        partial_commitments,
        {p: tuple(v) for p, v in transcript["partial_responses"].items()},
        transcript["challenge"])
    assert named == [3]


def test_several_bad_partials_are_all_named(key):
    shares = _shares(key)
    _, honest = joint_prove_opening(key, shares, QUORUM, context=b"ctx")
    faulty = {p: (tuple(honest["partial_responses"][p])[0] + 5,
                  tuple(honest["partial_responses"][p])[1])
              for p in (1, 3)}
    _, transcript = joint_prove_opening(key, shares, QUORUM, context=b"ctx",
                                        faulty=faulty)
    assert transcript["bad_partials"] == [1, 3]


def test_a_tampered_ladder_does_not_silently_exonerate(key):
    """Swapping the public ladder for another secret's makes everyone look bad."""
    shares, other = _shares(key), _shares(key, value=999)
    _, transcript = joint_prove_opening(key, shares, QUORUM, context=b"ctx")
    group = key.group
    partial_commitments = {
        p: group.decode(bytes.fromhex(v))
        for p, v in transcript["partial_commitments"].items()}
    named = audit_partials(
        key, other.coefficient_commitments, transcript["quorum"],
        partial_commitments,
        {p: tuple(v) for p, v in transcript["partial_responses"].items()},
        transcript["challenge"])
    assert named == QUORUM
