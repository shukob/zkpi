"""The values the circuit consumed are the values that were committed.

`policy_audit` proves the shares the nodes hold open to the committed policy,
and says plainly what it does not reach: those are not the shares MP-SPDZ
consumes, because MP-SPDZ works over its own prime field. It names two ways to
close that --- run the computation over the commitment field, or link the two
with a commit-and-prove argument. This is the second one.

Running over the commitment field costs 2.0 to 2.5 times the wall clock and
seven to fourteen times the traffic, on every quote, forever
(`artifacts/matched_field.json`). This costs one opening.

The check is one random linear combination. The dealer has already published a
commitment per input. Once the inputs are fixed, public coefficients are derived
from those commitments, the circuit computes

    s = sum_j c_j v_j + r

for a mask r the dealer also committed to, and opens s. Pedersen commitments are
additively homomorphic, so the same coefficients combine the commitments into one
that s must open. A node that feeds the circuit v_j + e_j instead of v_j shifts s
by sum_j c_j e_j, and for that to vanish the coefficients would have to satisfy
an equation the substituting node could not see when it chose e --- so it passes
with probability about 2^-CHALLENGE_BITS.

Two things this rests on, both checked rather than assumed.

**The fields must not both reduce.** The whole argument is that the same integer
appears on both sides, so the opened value has to stay below both the MPC prime
and the group order.

**The mask is not optional.** Without it each quote opens one linear equation in
the policy, and enough quotes with fresh coefficients solve for it.

**And those two together are what this costs, which is more than it first
looks.** The opening at 31-bit values, 40-bit coefficients and 166 inputs is 120
bits, which fits a 127-bit prime with seven bits to spare --- but the *mask* is
119 bits of that, and the mask is an input like any other, so it has to be dealt
to the nodes the way `qomm_transport.roles.split` deals everything: additively
over the integers, with `SLACK_BITS` of statistical room per share. That spends
the forty bits twice, once to hide the combination and once to hide each share,
and seven shares of a 119-bit value need **164 bits of field**. The 127-bit
prime does not hold it at any coefficient width --- not even at three bits, where
the check would be worthless anyway.

So the check does not run in the field it was proposed to save. It needs about
164 bits, against 253 for the group order, and at 253 the same widening also
makes `threshold_sigma` assemble correctly. **Whether it is worth widening to 164
rather than 253 is a real question and not an obvious one**, which is why
`check_width` takes the sharing into account and refuses rather than letting a
configuration through that would wrap.

What this does *not* do: it says the inputs were the committed ones, not that the
computation on them was right. That is what the malicious protocol is for, and
the two are complementary rather than alternatives.
"""

from __future__ import annotations

import hashlib
import math
import secrets
from dataclasses import dataclass
from typing import Any, Sequence

from .commit import Pedersen
from .groups import DOMAIN, Group
from .scheme import CommitmentScheme, PedersenScheme

CHALLENGE_BITS = 40
STATISTICAL_BITS = 40

# What fits a 127-bit prime, which is where the check has to run if the field is
# not being widened for anything else. The budget there is
# `challenge_bits + statistical_bits <= 41`, so soundness is bought back by
# repetition instead of by wider coefficients --- four independent combinations
# at ten bits each are 2^-40, the same figure `roles.SLACK_BITS` uses, and they
# open in one round because they do not depend on each other.
#
# The hiding cannot be bought back the same way, and `narrow_tradeoff` is where
# that is shown rather than asserted. Spending the budget on the gap leaves
# narrower coefficients, narrower coefficients need more repetitions, and
# repetitions dilute the gap again --- so the curve has a peak, at about
# **2^-34** near two to four coefficient bits. The stack's usual 2^-40 is not
# reachable at 127 bits at any point on it. That is the honest cost of running
# the check in the narrow field.
NARROW_CHALLENGE_BITS = 6
NARROW_STATISTICAL_BITS = 35
NARROW_REPEATS = 7


class WidthError(ValueError):
    """Raised when the opening would reduce in one of the two fields."""


def narrow_tradeoff(n_inputs: int, value_bits: int, field_bits: int = 127,
                    soundness_bits: int = 40, n_nodes: int = 7,
                    share_slack: int = 40) -> list[dict]:
    """Every coefficient width the narrow field allows, and what it costs.

    Soundness per round is at most `1 / (2^c - 1)`, so reaching a target takes
    `ceil(target / log2(2^c - 1))` rounds; repeating also dilutes the hiding by
    the number of rounds. The point of tabulating it is that "2^-40 hiding is
    not reachable at 127 bits" is false --- it is reachable, and what it costs
    is openings.
    """
    out = []
    for challenge_bits in range(2, 41):
        gap = field_bits - 1 - value_bits - max(0, (n_inputs - 1).bit_length()) \
            - share_slack - max(0, (n_nodes - 1).bit_length()) - 2 - challenge_bits
        if gap < 1:
            break
        per_round = math.log2((1 << challenge_bits) - 1)
        repeats = math.ceil(soundness_bits / per_round)
        out.append({"challenge_bits": challenge_bits, "statistical_bits": gap,
                    "repeats": repeats,
                    "soundness_bits": round(repeats * per_round, 1),
                    "hiding_bits": round(gap - math.log2(repeats), 1),
                    "openings": repeats, "masks_dealt": repeats})
    return out


def opening_bits(n_inputs: int, value_bits: int,
                 challenge_bits: int = CHALLENGE_BITS,
                 statistical_bits: int = STATISTICAL_BITS) -> int:
    """How wide the opened value can get, before any modulus is applied."""
    if n_inputs < 1:
        raise ValueError("an input check over no inputs checks nothing")
    combination = value_bits + challenge_bits + max(0, (n_inputs - 1).bit_length())
    return combination + statistical_bits + 1          # mask, and the carry


def mask_bits(n_inputs: int, value_bits: int,
              challenge_bits: int = CHALLENGE_BITS,
              statistical_bits: int = STATISTICAL_BITS) -> int:
    """How wide the mask has to be to hide the combination it is added to."""
    combination = value_bits + challenge_bits + max(0, (n_inputs - 1).bit_length())
    return combination + statistical_bits


def field_bits_needed(n_inputs: int, value_bits: int, n_nodes: int = 7,
                      challenge_bits: int = CHALLENGE_BITS,
                      statistical_bits: int = STATISTICAL_BITS,
                      share_slack: int = 40) -> int:
    """The field the whole check needs, mask and its shares included.

    The opening is the small half. The mask is an input like any other, so it is
    dealt additively over the integers with `share_slack` bits of room per share,
    and that is what sets the floor.
    """
    mask = mask_bits(n_inputs, value_bits, challenge_bits, statistical_bits)
    return mask + share_slack + max(0, (n_nodes - 1).bit_length()) + 2


def check_width(n_inputs: int, value_bits: int, mpc_prime_bits: int,
                group_order_bits: int, challenge_bits: int = CHALLENGE_BITS,
                statistical_bits: int = STATISTICAL_BITS, n_nodes: int = 7,
                share_slack: int = 40) -> int:
    """Refuse a configuration that would wrap in either field.

    Returns the field width the configuration needs, so a caller can record it.
    Counting only the opening --- which the first version of this did --- says a
    127-bit prime is enough, and it is not: the mask has to be shared too.
    """
    opening = opening_bits(n_inputs, value_bits, challenge_bits, statistical_bits)
    needed = field_bits_needed(n_inputs, value_bits, n_nodes, challenge_bits,
                               statistical_bits, share_slack)
    narrower = min(mpc_prime_bits, group_order_bits)
    if needed >= narrower:
        raise WidthError(
            f"{n_inputs} inputs of {value_bits} bits with {challenge_bits}-bit "
            f"coefficients open to {opening} bits, which fits --- but the "
            f"{mask_bits(n_inputs, value_bits, challenge_bits, statistical_bits)}"
            f"-bit mask has to be dealt to {n_nodes} nodes with {share_slack} "
            f"bits of slack per share, and that needs {needed} bits against the "
            f"narrower of the MPC prime ({mpc_prime_bits}) and the group order "
            f"({group_order_bits}). The forty bits are spent twice, once on the "
            f"combination and once on each share. Widen the field, or say in the "
            f"artifact which of the two hidings was given up.")
    return needed


def as_scheme(key) -> CommitmentScheme:
    """Accept either a commitment scheme or the Pedersen key this used to take."""
    return key if isinstance(key, CommitmentScheme) else PedersenScheme(key)


def coefficients(scheme, commitments: Sequence[Any], mask_commitment: Any,
                 context: bytes, challenge_bits: int = CHALLENGE_BITS,
                 round_index: int = 0) -> list[int]:
    """Public coefficients, derived from the commitments and nothing else.

    Deriving them from the commitments is what puts them after the inputs: a
    node that wants sum c_j e_j to vanish has to choose e before it can see c,
    and changing e changes nothing about c because c does not depend on e.
    """
    scheme = as_scheme(scheme)
    seed = hashlib.sha512(DOMAIN + b":input-check:v1")
    seed.update(len(context).to_bytes(4, "big"))
    seed.update(context)
    seed.update(len(commitments).to_bytes(4, "big"))
    for commitment in commitments:
        encoded = scheme.encode(commitment)
        seed.update(len(encoded).to_bytes(4, "big"))
        seed.update(encoded)
    encoded = scheme.encode(mask_commitment)
    seed.update(len(encoded).to_bytes(4, "big"))
    seed.update(encoded)
    seed.update(round_index.to_bytes(4, "big"))
    root = seed.digest()

    out, span = [], 1 << challenge_bits
    for index in range(len(commitments)):
        digest = hashlib.sha512(root + index.to_bytes(4, "big")).digest()
        # a coefficient of zero would leave that input unchecked
        out.append(1 + int.from_bytes(digest, "big") % (span - 1))
    return out


@dataclass(frozen=True)
class InputCheck:
    """What is published so anyone can check the inputs were the committed ones.

    One entry per repetition. They are independent, so they open together in one
    round, and their soundness multiplies.
    """

    commitments: list
    mask_commitments: list
    openings: list
    opening_blindings: list
    challenge_bits: int = CHALLENGE_BITS

    def soundness_bits(self) -> int:
        return self.challenge_bits * len(self.openings)

    @property
    def repeats(self) -> int:
        return len(self.openings)


def sample_mask(n_inputs: int, value_bits: int,
                challenge_bits: int = CHALLENGE_BITS,
                statistical_bits: int = STATISTICAL_BITS, rng=None) -> int:
    """A mask wide enough that the opening hides the combination."""
    combination = value_bits + challenge_bits + max(0, (n_inputs - 1).bit_length())
    rng = rng or secrets.SystemRandom()
    return rng.randrange(1 << (combination + statistical_bits))


def build(key, values: Sequence[int], blindings: Sequence[int],
          context: bytes, challenge_bits: int = CHALLENGE_BITS,
          statistical_bits: int = STATISTICAL_BITS, repeats: int = 1,
          value_bits: int = 32, masks: Sequence[int] | None = None,
          mask_blindings: Sequence[int] | None = None) -> InputCheck:
    """The dealer's side: commit, derive, combine, open.

    In deployment the circuit computes the combination from shares --- public
    coefficient times secret share is local, so it costs no communication --- and
    the opening is the one round this check adds. Here the values are in hand,
    which is what makes the test able to substitute one.
    """
    scheme = as_scheme(key)
    if len(values) != len(blindings):
        raise ValueError("every value needs its blinding")
    if repeats < 1:
        raise ValueError("a check with no repetitions checks nothing")
    masks = list(masks) if masks is not None else [
        sample_mask(len(values), value_bits, challenge_bits, statistical_bits)
        for _ in range(repeats)]
    mask_blindings = list(mask_blindings) if mask_blindings is not None else [
        scheme.random_blinding() for _ in range(repeats)]

    commitments = [scheme.commit(v, r) for v, r in zip(values, blindings)]
    mask_commitments = [scheme.commit(m, b) for m, b in zip(masks, mask_blindings)]

    openings, opening_blindings = [], []
    for index in range(repeats):
        c = coefficients(scheme, commitments, mask_commitments[index], context,
                         challenge_bits, round_index=index)
        openings.append(sum(cj * v for cj, v in zip(c, values)) + masks[index])
        opening_blindings.append(
            sum(cj * r for cj, r in zip(c, blindings)) + mask_blindings[index])
    return InputCheck(commitments, mask_commitments, openings, opening_blindings,
                      challenge_bits)


def verify(key, check: InputCheck, context: bytes) -> tuple[bool, str]:
    """Anyone's side: rederive the coefficients and combine the commitments."""
    scheme = as_scheme(key)
    if not check.commitments:
        return False, "the check covers no inputs"
    for index in range(check.repeats):
        c = coefficients(scheme, check.commitments, check.mask_commitments[index],
                         context, check.challenge_bits, round_index=index)
        combined = check.mask_commitments[index]
        for coefficient, commitment in zip(c, check.commitments):
            combined = scheme.add(combined, scheme.scale(commitment, coefficient))
        if not scheme.equal(scheme.commit(check.openings[index],
                                          check.opening_blindings[index]),
                            combined):
            return False, (f"combination {index} is not what the committed inputs "
                           f"combine to: an input the circuit used was not the "
                           f"one that was committed")
    return True, "ok"
