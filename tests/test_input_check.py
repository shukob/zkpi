"""The inputs the circuit consumed are the ones that were committed.

The alternative to this check is running the whole computation over the
commitment field, which `artifacts/matched_field.json` prices at 2.0 to 2.5
times the wall clock and seven to fourteen times the traffic. So these tests are
mostly about the ways the cheap version could fail to be a check at all.
"""

from __future__ import annotations

import pytest

from zk.commit import Pedersen
from zk.groups import make_group
from zk.input_check import (CHALLENGE_BITS, NARROW_CHALLENGE_BITS,
                            NARROW_REPEATS, NARROW_STATISTICAL_BITS,
                            InputCheck, WidthError, build, check_width,
                            coefficients, field_bits_needed, mask_bits,
                            narrow_tradeoff, opening_bits, verify)

CONTEXT = b"qomm:test:slot:7"


@pytest.fixture(scope="module")
def key() -> Pedersen:
    return Pedersen(make_group("ed25519"), b"qomm:input-check:test")


def policy(key: Pedersen, n: int = 24) -> tuple[list[int], list[int]]:
    # the ranges the DSL declares: offsets and coefficients, signs included
    values = [(-1) ** i * (37 * i + 5) for i in range(n)]
    return values, [key.random_blinding() for _ in values]


def test_honest_inputs_verify(key: Pedersen) -> None:
    values, blindings = policy(key)
    check = build(key, values, blindings, CONTEXT)
    assert verify(key, check, CONTEXT) == (True, "ok")


@pytest.mark.parametrize("position", (0, 7, 23))
@pytest.mark.parametrize("error", (1, -1, 1000, -4096))
def test_one_substituted_input_is_caught(key: Pedersen, position: int,
                                         error: int) -> None:
    """A node feeds the circuit something other than the share it was dealt."""
    values, blindings = policy(key)
    honest = build(key, values, blindings, CONTEXT)

    # the commitments stand --- the dealer published them --- but the circuit
    # ran on a different value, so the opening is the shifted one
    substituted = list(values)
    substituted[position] += error
    lying = build(key, substituted, blindings, CONTEXT)
    forged = InputCheck(honest.commitments, honest.mask_commitments,
                        lying.openings, honest.opening_blindings)

    ok, reason = verify(key, forged, CONTEXT)
    assert not ok and "was not the one that was committed" in reason


def test_errors_across_two_inputs_do_not_cancel(key: Pedersen) -> None:
    """The obvious way to try to survive: shift one input up and another down."""
    values, blindings = policy(key)
    honest = build(key, values, blindings, CONTEXT)
    substituted = list(values)
    substituted[3] += 500
    substituted[11] -= 500          # cancels only if c_3 happens to equal c_11
    lying = build(key, substituted, blindings, CONTEXT)
    forged = InputCheck(honest.commitments, honest.mask_commitments,
                        lying.openings, honest.opening_blindings)
    assert not verify(key, forged, CONTEXT)[0]


def test_the_coefficients_come_from_the_commitments(key: Pedersen) -> None:
    """Which is what puts them after the inputs rather than before."""
    values, blindings = policy(key)
    first = build(key, values, blindings, CONTEXT)
    moved = list(values)
    moved[5] += 1
    second = build(key, moved, blindings, CONTEXT)
    a = coefficients(key.group, first.commitments, first.mask_commitments[0], CONTEXT)
    b = coefficients(key.group, second.commitments, second.mask_commitments[0], CONTEXT)
    assert a != b, ("the coefficients did not move with the commitments, so a "
                    "node could choose its error after seeing them")


def test_the_context_separates_slots(key: Pedersen) -> None:
    values, blindings = policy(key)
    check = build(key, values, blindings, CONTEXT)
    assert verify(key, check, CONTEXT)[0]
    assert not verify(key, check, b"qomm:test:slot:8")[0]


def test_no_coefficient_is_zero(key: Pedersen) -> None:
    """A zero would leave that input unchecked, which is the quiet failure."""
    values, blindings = policy(key, n=64)
    check = build(key, values, blindings, CONTEXT)
    c = coefficients(key.group, check.commitments, check.mask_commitments[0], CONTEXT)
    assert len(c) == 64 and all(x > 0 for x in c)
    assert all(x < (1 << CHALLENGE_BITS) for x in c)


def test_the_mask_moves_the_opening(key: Pedersen) -> None:
    """Same policy twice: the openings must not be the same number."""
    values, blindings = policy(key)
    openings = {build(key, values, blindings, CONTEXT).openings[0] for _ in range(8)}
    assert len(openings) == 8, ("the opening repeated, so a policy priced twice "
                                "would leak the same equation twice")


# ---- the width budget, which is the whole reason this works across two fields

def test_the_shipped_field_does_not_hold_this_check(key: Pedersen) -> None:
    """The finding that came out of implementing it, kept as a test.

    Counting only the opening says the 127-bit prime is enough --- 120 bits with
    seven to spare. It is not. The mask is 119 of those 120 bits and it is an
    input like any other, so it is dealt with forty bits of slack per share, and
    seven shares of it need 164 bits.
    """
    with pytest.raises(WidthError, match="spent twice"):
        check_width(n_inputs=166, value_bits=31, mpc_prime_bits=127,
                    group_order_bits=252)


@pytest.mark.parametrize("challenge_bits", (3, 8, 16, 32, 40))
def test_no_coefficient_width_rescues_the_127_bit_field(key: Pedersen,
                                                        challenge_bits: int) -> None:
    """Narrowing the coefficients trades soundness away and still does not fit."""
    with pytest.raises(WidthError):
        check_width(n_inputs=166, value_bits=31, mpc_prime_bits=127,
                    group_order_bits=252, challenge_bits=challenge_bits)


def test_a_wide_enough_field_is_accepted(key: Pedersen) -> None:
    needed = field_bits_needed(166, 31)
    assert needed == 164, needed
    assert check_width(n_inputs=166, value_bits=31, mpc_prime_bits=192,
                       group_order_bits=252) == 164
    # and the group order, which is where the sigma assembly wants it anyway
    assert check_width(n_inputs=166, value_bits=31, mpc_prime_bits=252,
                       group_order_bits=252) == 164


def test_the_number_of_inputs_is_almost_free(key: Pedersen) -> None:
    """Which is the good news, and the reason the check is worth having at all.

    The width grows with log2 of the input count, so covering a thousand times
    more inputs costs ten bits. What sets the floor is the mask and its slack,
    not how much the check covers --- so one check over everything is the right
    shape, rather than one per maker.
    """
    small = field_bits_needed(16, 31)
    shipped = field_bits_needed(166, 31)
    huge = field_bits_needed(1 << 30, 31)
    assert small == 160 and shipped == 164 and huge == 186
    assert huge - small == 26, "the growth stopped being logarithmic"
    # a million-fold increase in coverage still fits the group order
    check_width(n_inputs=1 << 30, value_bits=31, mpc_prime_bits=252,
                group_order_bits=252)


def test_the_opening_stays_inside_the_narrower_field(key: Pedersen) -> None:
    """Not just the bound --- the number an honest run actually produces."""
    values, blindings = policy(key, n=166)
    check = build(key, values, blindings, CONTEXT)
    assert all(0 < o < (1 << opening_bits(166, 31)) for o in check.openings)
    assert all(o < (1 << 127) for o in check.openings), \
        "the opening would reduce in the MPC prime"


# ---- the narrow field, which is where the check has to run if nothing else
#      is widening it

NARROW = dict(challenge_bits=NARROW_CHALLENGE_BITS,
              statistical_bits=NARROW_STATISTICAL_BITS)


def test_the_narrow_configuration_fits_the_default_field(key: Pedersen) -> None:
    """Six-bit coefficients and a 35-bit gap: 126 bits against a 127-bit prime."""
    needed = check_width(n_inputs=166, value_bits=32, mpc_prime_bits=127,
                         group_order_bits=252, **NARROW)
    assert needed == 126, needed


def test_repetition_buys_the_soundness_back(key: Pedersen) -> None:
    values, blindings = policy(key, n=166)
    check = build(key, values, blindings, CONTEXT, repeats=NARROW_REPEATS, **NARROW)
    assert check.repeats == 7
    assert check.soundness_bits() == 42, (
        "seven six-bit combinations should clear the 2^-40 the rest of the "
        "stack uses")
    assert verify(key, check, CONTEXT) == (True, "ok")


@pytest.mark.parametrize("position", (0, 83, 165))
def test_the_narrow_check_still_catches_a_substitution(key: Pedersen,
                                                       position: int) -> None:
    values, blindings = policy(key, n=166)
    honest = build(key, values, blindings, CONTEXT, repeats=NARROW_REPEATS, **NARROW)
    substituted = list(values)
    substituted[position] += 17
    lying = build(key, substituted, blindings, CONTEXT, repeats=NARROW_REPEATS,
                  masks=[0] * NARROW_REPEATS, **NARROW)
    forged = InputCheck(honest.commitments, honest.mask_commitments,
                        lying.openings, honest.opening_blindings)
    assert not verify(key, forged, CONTEXT)[0]


def test_every_repetition_uses_different_coefficients(key: Pedersen) -> None:
    """Otherwise repeating buys nothing: the same test four times is one test."""
    values, blindings = policy(key, n=40)
    check = build(key, values, blindings, CONTEXT, repeats=4, **NARROW)
    rounds = [tuple(coefficients(key.group, check.commitments,
                                 check.mask_commitments[i], CONTEXT,
                                 NARROW_CHALLENGE_BITS, round_index=i))
              for i in range(4)]
    assert len(set(rounds)) == 4


def test_forty_bit_hiding_is_not_reachable_in_the_narrow_field(key: Pedersen) -> None:
    """Repetition buys soundness back and cannot buy the hiding back.

    Every coefficient width the field allows, with the repetitions each needs to
    reach 2^-40 soundness, and the hiding that survives the dilution. The peak is
    about 2^-34, so the stack's usual 2^-40 is out of reach at 127 bits --- which
    is the cost of the narrow field, and it is a fact about the budget rather
    than about this implementation.
    """
    curve = narrow_tradeoff(166, 32)
    assert curve, "the field allows no coefficient width at all"
    best = max(row["hiding_bits"] for row in curve)
    assert 33 < best < 35, best
    assert best < 40, "2^-40 hiding turned out to be reachable after all"
    # and the peak is at narrow coefficients with many repetitions, not wide ones
    peak = max(curve, key=lambda row: row["hiding_bits"])
    assert peak["challenge_bits"] <= 4 and peak["repeats"] >= 11
