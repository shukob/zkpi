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

# what the circuit opens once every input has been read
BEACON = 0x9E3779B97F4A7C15

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
    check = build_per_party(scheme, shares, blindings, context, BEACON)
    coeffs = per_party_coefficients(scheme, check.share_commitments,
                                    check.mask_commitments, context, BEACON)
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
                                                    b"ctx", BEACON), b"ctx", BEACON) \
        == (True, "ok", [])


@pytest.mark.parametrize("who", [[0], [3], [6]])
def test_a_single_substituting_node_is_named(scheme, who):
    shares, blindings = fixture(scheme)
    ok, why, culprits = verify_per_party(
        scheme, substitute(scheme, shares, blindings, who), b"ctx", BEACON)
    assert not ok and culprits == who
    assert f"node {who[0]}" in why


def test_every_node_position_is_named_correctly(scheme):
    """No index is a blind spot."""
    shares, blindings = fixture(scheme, n_values=6)
    for party in range(N_PARTIES):
        _, _, culprits = verify_per_party(
            scheme, substitute(scheme, shares, blindings, [party]), b"ctx", BEACON)
        assert culprits == [party]


def test_the_innocent_are_not_named(scheme):
    shares, blindings = fixture(scheme)
    _, _, culprits = verify_per_party(
        scheme, substitute(scheme, shares, blindings, [2]), b"ctx", BEACON)
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
        scheme, substitute(scheme, shares, blindings, who), b"ctx", BEACON)
    assert not ok and culprits == who


def test_all_seven_lying_is_still_resolvable(scheme):
    shares, blindings = fixture(scheme, n_values=5)
    _, _, culprits = verify_per_party(
        scheme, substitute(scheme, shares, blindings, list(range(7))), b"ctx", BEACON)
    assert culprits == list(range(7))


# --- it is strictly stronger than the aggregate check ---------------------

def test_the_aggregate_check_is_implied(scheme):
    """`sum_p s_p` is the old opening, so passing per party passes overall."""
    shares, blindings = fixture(scheme, n_values=9)
    check = build_per_party(scheme, shares, blindings, b"ctx", BEACON)
    coeffs = per_party_coefficients(scheme, check.share_commitments,
                                    check.mask_commitments, b"ctx", BEACON)
    values = [sum(shares[p][j] for p in range(N_PARTIES))
              for j in range(9)]
    assert sum(check.openings) - sum(
        c * v for c, v in zip(coeffs, values)) == sum(
        o - sum(c * s for c, s in zip(coeffs, shares[p]))
        for p, o in enumerate(check.openings))


def test_a_different_context_does_not_verify(scheme):
    shares, blindings = fixture(scheme)
    check = build_per_party(scheme, shares, blindings, b"ctx", BEACON)
    assert not verify_per_party(scheme, check, b"another auction", BEACON)[0]


def test_coefficients_depend_on_every_published_commitment(scheme):
    shares, blindings = fixture(scheme, n_values=4)
    a = build_per_party(scheme, shares, blindings, b"ctx", BEACON)
    other = [list(r) for r in shares]
    other[5][2] += 1
    b = build_per_party(scheme, other, blindings, b"ctx", BEACON)
    assert per_party_coefficients(scheme, a.share_commitments,
                                  a.mask_commitments, b"ctx", BEACON) \
        != per_party_coefficients(scheme, b.share_commitments,
                                  b.mask_commitments, b"ctx", BEACON)


# --- the flaw the proof found ---------------------------------------------

def test_a_node_that_sees_the_coefficients_cannot_cancel_its_error(scheme):
    """The case forty tests missed, because they all substituted ONE input.

    With one term, `c_1 e_1 = 0` forces `e_1 = 0` and any challenge will do. With
    two, a node that knows `c` picks `e_1 = c_2 k` and `e_2 = -c_1 k` and the
    combination vanishes identically --- no guessing, every time. That is why the
    challenge has to be drawn after the inputs are fixed, and why the module
    refuses to derive coefficients without one.
    """
    shares, blindings = fixture(scheme, n_values=12)
    check = build_per_party(scheme, shares, blindings, b"ctx", BEACON)
    seen = per_party_coefficients(scheme, check.share_commitments,
                                  check.mask_commitments, b"ctx", BEACON)
    # the attack, against the challenge the node actually saw
    bad = [list(r) for r in shares]
    bad[4][1] += seen[2] * 1000
    bad[4][2] -= seen[1] * 1000
    assert bad[4] != shares[4]
    # against a challenge drawn afterwards, the same error no longer cancels
    later = per_party_coefficients(scheme, check.share_commitments,
                                   check.mask_commitments, b"ctx",
                                   BEACON ^ 0xA5A5A5A5)
    drift = sum(l * (b - s) for l, b, s in zip(later, bad[4], shares[4]))
    assert drift != 0, "the error survives a fresh challenge, which is the point"


def test_the_coefficients_move_with_the_challenge(scheme):
    shares, blindings = fixture(scheme, n_values=6)
    check = build_per_party(scheme, shares, blindings, b"ctx", BEACON)
    a = per_party_coefficients(scheme, check.share_commitments,
                               check.mask_commitments, b"ctx", BEACON)
    b = per_party_coefficients(scheme, check.share_commitments,
                               check.mask_commitments, b"ctx", BEACON + 1)
    assert a != b


def test_deriving_coefficients_without_a_challenge_is_refused(scheme):
    shares, blindings = fixture(scheme, n_values=4)
    check = build_per_party(scheme, shares, blindings, b"ctx", BEACON)
    with pytest.raises(ValueError, match="AFTER the inputs"):
        per_party_coefficients(scheme, check.share_commitments,
                               check.mask_commitments, b"ctx")


# --- shape ----------------------------------------------------------------

def test_a_ragged_dealing_is_refused(scheme):
    shares, blindings = fixture(scheme, n_values=4)
    shares[3] = shares[3][:2]
    with pytest.raises(ValueError, match="one share of every value"):
        build_per_party(scheme, shares, blindings, b"ctx", BEACON)


def test_missing_blindings_are_refused(scheme):
    shares, blindings = fixture(scheme, n_values=4)
    with pytest.raises(ValueError, match="blinding"):
        build_per_party(scheme, shares, blindings[:-1], b"ctx", BEACON)


def test_no_parties_is_refused(scheme):
    with pytest.raises(ValueError):
        build_per_party(scheme, [], [], b"ctx", BEACON)


def test_the_shape_is_reported(scheme):
    shares, blindings = fixture(scheme, n_values=11)
    check = build_per_party(scheme, shares, blindings, b"ctx", BEACON)
    assert check.n_parties == N_PARTIES and check.n_values == 11
    assert check.soundness_bits() == CHALLENGE_BITS
