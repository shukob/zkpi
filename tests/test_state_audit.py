"""The state audit has to reject, not merely accept.

A chain that verifies when it is correct proves nothing on its own; the claim is
that it *fails* when the maker misreports, and there are three separate ways to
misreport that a venue would respond to differently. Each is injected here.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zk.groups import make_group                                    # noqa: E402
from zk.state_audit import StateAuditor                             # noqa: E402


@pytest.fixture(scope="module")
def auditor():
    return StateAuditor(make_group("ed25519"), ceiling=1 << 12)


def _limit(auditor, value=400):
    blinding = auditor.key.random_blinding()
    return auditor.commit_limit(value, blinding), blinding


def _chain(auditor, fills, *, start=0, limit_value=400):
    """Run a maker honestly through a sequence of fills."""
    limit, limit_blinding = _limit(auditor, limit_value)
    blinding = auditor.key.random_blinding()
    opening = auditor.key.commit(start, blinding)
    inventory, steps = start, []
    for index, filled in enumerate(fills):
        new_blinding = auditor.key.random_blinding()
        step, inventory = auditor.prove_update(
            step=index, old_inventory=inventory, old_blinding=blinding,
            filled=filled, fill_blinding=auditor.key.random_blinding(),
            limit=limit_value, limit_blinding=limit_blinding,
            new_blinding=new_blinding)
        steps.append(step)
        blinding = new_blinding
    return opening, steps, limit


def test_an_honest_chain_verifies(auditor):
    opening, steps, limit = _chain(auditor, [50, -20, 30, -70])
    ok, reason = auditor.verify_chain(opening, steps, limit)
    assert ok, reason


def test_the_limit_itself_is_checked(auditor):
    """A limit outside the public ceiling is not a limit anyone can rely on."""
    with pytest.raises(ValueError):
        auditor.commit_limit(auditor.ceiling + 1, auditor.key.random_blinding())


def test_inventory_that_did_not_move_is_rejected(auditor):
    """The commonest misreport: take the fill, leave the book where it was."""
    limit_value = 400
    limit, limit_blinding = _limit(auditor, limit_value)
    blinding = auditor.key.random_blinding()
    opening = auditor.key.commit(0, blinding)
    # claim the inventory is still zero after filling 50
    honest, _ = auditor.prove_update(
        step=0, old_inventory=0, old_blinding=blinding, filled=50,
        fill_blinding=auditor.key.random_blinding(), limit=limit_value,
        limit_blinding=limit_blinding, new_blinding=auditor.key.random_blinding())
    frozen, _ = auditor.prove_update(
        step=0, old_inventory=0, old_blinding=blinding, filled=0,
        fill_blinding=auditor.key.random_blinding(), limit=limit_value,
        limit_blinding=limit_blinding, new_blinding=auditor.key.random_blinding())
    # splice the unmoved inventory onto the real fill
    forged = type(honest)(step=0, inventory=frozen.inventory, fill=honest.fill,
                          follows=honest.follows, arithmetic=honest.arithmetic,
                          below_cap=frozen.below_cap, above_floor=frozen.above_floor)
    ok, reason = auditor.verify_chain(opening, [forged], limit)
    assert not ok and "proofs" in reason


def test_the_wrong_sign_is_rejected(auditor):
    """Moving the book the wrong way would hide a position rather than carry it."""
    limit_value = 400
    limit, limit_blinding = _limit(auditor, limit_value)
    blinding = auditor.key.random_blinding()
    opening = auditor.key.commit(100, blinding)
    step, _ = auditor.prove_update(
        step=0, old_inventory=100, old_blinding=blinding, filled=-40,
        fill_blinding=auditor.key.random_blinding(), limit=limit_value,
        limit_blinding=limit_blinding, new_blinding=auditor.key.random_blinding())
    # the honest result is 140; present the same proofs against a claim of 60
    wrong_blinding = auditor.key.random_blinding()
    forged = type(step)(step=0, inventory=auditor.key.commit(60, wrong_blinding),
                        fill=step.fill, follows=step.follows,
                        arithmetic=step.arithmetic, below_cap=step.below_cap,
                        above_floor=step.above_floor)
    ok, _ = auditor.verify_chain(opening, [forged], limit)
    assert not ok


def test_a_breach_cannot_be_proved_at_all(auditor):
    """A maker past its own limit has nothing to present, which is the point."""
    limit_value = 100
    limit, limit_blinding = _limit(auditor, limit_value)
    blinding = auditor.key.random_blinding()
    with pytest.raises(ValueError, match="committed limit"):
        auditor.prove_update(
            step=0, old_inventory=90, old_blinding=blinding, filled=-50,
            fill_blinding=auditor.key.random_blinding(), limit=limit_value,
            limit_blinding=limit_blinding, new_blinding=auditor.key.random_blinding())


def test_a_replayed_state_breaks_the_chain(auditor):
    """Running the book back to an earlier state is equivocation, not arithmetic."""
    opening, steps, limit = _chain(auditor, [50, -20, 30])
    replayed = [steps[0], steps[2]]          # skip the middle, so step 2 follows nothing
    ok, reason = auditor.verify_chain(opening, replayed, limit)
    assert not ok and "follow" in reason


def test_two_parallel_states_break_the_chain(auditor):
    """One book, not two: a fork is exactly what `follows` is there to catch."""
    opening, steps, limit = _chain(auditor, [50, -20])
    other_opening, other_steps, _ = _chain(auditor, [10, 10])
    forked = [steps[0], other_steps[1]]
    ok, reason = auditor.verify_chain(opening, forked, limit)
    assert not ok and "follow" in reason


def test_the_limit_commitment_cannot_be_swapped_for_a_looser_one(auditor):
    """Tightening or loosening the promise after the fact is not available."""
    opening, steps, limit = _chain(auditor, [50, -20], limit_value=400)
    looser, _ = _limit(auditor, 4000)
    ok, _ = auditor.verify_chain(opening, steps, looser)
    assert not ok


def test_a_chain_of_many_steps_stays_verifiable(auditor):
    fills = [17, -33, 41, -8, 25, -60, 12, 9, -14, 30]
    opening, steps, limit = _chain(auditor, fills, limit_value=400)
    ok, reason = auditor.verify_chain(opening, steps, limit)
    assert ok, reason
    assert len(steps) == len(fills)
