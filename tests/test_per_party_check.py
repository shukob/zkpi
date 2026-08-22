"""Naming the node that fed the circuit something other than what it was dealt.

`input_check` proves that *an* input was substituted. This proves *which node*
did it, and the difference is the difference between the first and the fourth of
the five rungs in `ACCOUNTABILITY.md`.

The claim being tested is not that a linear check works --- that is
`test_input_check.py` --- but three things specific to doing it per party:

  * the culprits are named, and only the culprits
  * there is **no capacity limit**, unlike the Reed--Solomon decode in
    `qomm_audit.locate`, because each party's check stands alone
  * it needs a **narrower** field than the aggregate check, because the mask is
    the party's own input and is not dealt across nodes
"""

from __future__ import annotations

import secrets

import pytest

from zk.commit import Pedersen
from zk.groups import make_group
from zk.input_check import (CHALLENGE_BITS, PerPartyCheck, build,
                            build_per_party, field_bits_needed,
                            per_party_coefficients, per_party_field_bits,
                            verify, verify_per_party)
from zk.scheme import PedersenScheme

N_PARTIES = 7
SHARE_BITS = 71                      # value_bits 31 + roles.SLACK_BITS 40


@pytest.fixture(scope="module")
def scheme():
    return PedersenScheme(Pedersen(make_group("ed25519"), b"qomm:pedersen:v1"))


def fixture(scheme, n_values=12, n_parties=N_PARTIES):
    shares = [[secrets.randbelow(1 << SHARE_BITS) for _ in range(n_values)]
              for _ in range(n_parties)]
    blindings = [[scheme.random_blinding() for _ in range(n_values)]
                 for _ in range(n_parties)]
    return shares, blindings


def substitute(scheme, shares, blindings, who: list[int], context=b"ctx"):
    """Build honestly, then open what a substituting node would have opened.

    Doing it this way rather than committing to the substituted value is the
    point: the commitments are what the dealer published and the node cannot
    change them. What it can change is the number it feeds the circuit.
    """
    check = build_per_party(scheme, shares, blindings, context)
    coeffs = per_party_coefficients(scheme, check.share_commitments,
                                    check.mask_commitments, context)
    openings = list(check.openings)
    for party in who:
        drift = sum(coeffs) or 1
        openings[party] += drift
    return PerPartyCheck(check.share_commitments, check.mask_commitments,
                         openings, check.opening_blindings, check.challenge_bits)


# --- the width, which is the surprising part ------------------------------

def test_the_per_party_check_needs_a_narrower_field():
    """160 against 164. The aggregate mask is dealt across nodes; this one is not."""
    assert per_party_field_bits(166, 31) == 160
    assert field_bits_needed(166, 31) == 164
    assert per_party_field_bits(166, 31) < field_bits_needed(166, 31)


def test_neither_fits_the_default_prime_and_both_fit_the_group_order():
    assert per_party_field_bits(166, 31) > 127
    assert per_party_field_bits(166, 31) < 252


@pytest.mark.parametrize("n_inputs", [1, 16, 166, 1024])
def test_width_grows_with_the_input_count_and_nothing_else(n_inputs):
    """It must not depend on the node count, which is the whole saving."""
    assert (per_party_field_bits(n_inputs, 31, n_nodes=3)
            == per_party_field_bits(n_inputs, 31, n_nodes=99))


# --- naming ---------------------------------------------------------------

def test_an_honest_dealing_names_nobody(scheme):
    shares, blindings = fixture(scheme)
    assert verify_per_party(scheme, build_per_party(scheme, shares, blindings,
                                                    b"ctx"), b"ctx") \
        == (True, "ok", [])


@pytest.mark.parametrize("who", [[0], [3], [6]])
def test_a_single_substituting_node_is_named(scheme, who):
    shares, blindings = fixture(scheme)
    ok, why, culprits = verify_per_party(
        scheme, substitute(scheme, shares, blindings, who), b"ctx")
    assert not ok and culprits == who
    assert f"node {who[0]}" in why


def test_every_node_position_is_named_correctly(scheme):
    """No index is a blind spot."""
    shares, blindings = fixture(scheme, n_values=6)
    for party in range(N_PARTIES):
        _, _, culprits = verify_per_party(
            scheme, substitute(scheme, shares, blindings, [party]), b"ctx")
        assert culprits == [party]


def test_the_innocent_are_not_named(scheme):
    shares, blindings = fixture(scheme)
    _, _, culprits = verify_per_party(
        scheme, substitute(scheme, shares, blindings, [2]), b"ctx")
    assert set(culprits) == {2}
    assert len(culprits) == 1


# --- no capacity limit, which is the difference from the decode -----------

@pytest.mark.parametrize("n_bad", [1, 2, 3, 4, 6, 7])
def test_there_is_no_threshold_on_how_many_can_be_named(scheme, n_bad):
    """`qomm_audit.locate` caps at T=2 because it decodes a code. This does not.

    Each party's check stands alone against that party's own published
    commitments, so five liars are as nameable as one --- which matters, because
    the case a decoder gives up on is exactly the case an operator most needs a
    name for.
    """
    shares, blindings = fixture(scheme, n_values=8)
    who = list(range(n_bad))
    ok, _, culprits = verify_per_party(
        scheme, substitute(scheme, shares, blindings, who), b"ctx")
    assert not ok and culprits == who


def test_all_seven_lying_is_still_resolvable(scheme):
    shares, blindings = fixture(scheme, n_values=5)
    _, _, culprits = verify_per_party(
        scheme, substitute(scheme, shares, blindings, list(range(7))), b"ctx")
    assert culprits == list(range(7))


# --- it is strictly stronger than the aggregate check ---------------------

def test_the_aggregate_check_is_implied(scheme):
    """`sum_p s_p` is the old opening, so passing per party passes overall."""
    shares, blindings = fixture(scheme, n_values=9)
    check = build_per_party(scheme, shares, blindings, b"ctx")
    coeffs = per_party_coefficients(scheme, check.share_commitments,
                                    check.mask_commitments, b"ctx")
    values = [sum(shares[p][j] for p in range(N_PARTIES))
              for j in range(9)]
    assert sum(check.openings) - sum(
        c * v for c, v in zip(coeffs, values)) == sum(
        o - sum(c * s for c, s in zip(coeffs, shares[p]))
        for p, o in enumerate(check.openings))


def test_a_different_context_does_not_verify(scheme):
    shares, blindings = fixture(scheme)
    check = build_per_party(scheme, shares, blindings, b"ctx")
    assert not verify_per_party(scheme, check, b"another auction")[0]


def test_coefficients_depend_on_every_published_commitment(scheme):
    shares, blindings = fixture(scheme, n_values=4)
    a = build_per_party(scheme, shares, blindings, b"ctx")
    other = [list(r) for r in shares]
    other[5][2] += 1
    b = build_per_party(scheme, other, blindings, b"ctx")
    assert per_party_coefficients(scheme, a.share_commitments,
                                  a.mask_commitments, b"ctx") \
        != per_party_coefficients(scheme, b.share_commitments,
                                  b.mask_commitments, b"ctx")


# --- shape ----------------------------------------------------------------

def test_a_ragged_dealing_is_refused(scheme):
    shares, blindings = fixture(scheme, n_values=4)
    shares[3] = shares[3][:2]
    with pytest.raises(ValueError, match="one share of every value"):
        build_per_party(scheme, shares, blindings, b"ctx")


def test_missing_blindings_are_refused(scheme):
    shares, blindings = fixture(scheme, n_values=4)
    with pytest.raises(ValueError, match="blinding"):
        build_per_party(scheme, shares, blindings[:-1], b"ctx")


def test_no_parties_is_refused(scheme):
    with pytest.raises(ValueError):
        build_per_party(scheme, [], [], b"ctx")


def test_the_shape_is_reported(scheme):
    shares, blindings = fixture(scheme, n_values=11)
    check = build_per_party(scheme, shares, blindings, b"ctx")
    assert check.n_parties == N_PARTIES and check.n_values == 11
    assert check.soundness_bits() == CHALLENGE_BITS
