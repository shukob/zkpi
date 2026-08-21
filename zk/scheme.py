"""The commitment layer as a choice, so a different one can be measured.

`groups.py` already made the *group* pluggable, which covers every
discrete-logarithm scheme and no others. VOLE-based commitments are not group
elements at all --- a commitment is an information-theoretic MAC, and the
homomorphic operations are field arithmetic rather than scalar multiplications
--- so the seam has to move up from the group to the commitment.

    commit(value, blinding)     ->  Commitment
    add(a, b)                   ->  Commitment      (was group.mul)
    scale(c, k)                 ->  Commitment      (was group.point_pow)
    negate(c), zero(), encode(c), equal(a, b)
    scalar_modulus, random_scalar(), random_blinding()

Two implementations, and **they do not promise the same thing**, which is the
point of writing the difference down rather than hiding it behind an interface:

`PedersenScheme` is *publicly verifiable*. A commitment is a group element that
anybody can hold, combine and check, which is what `threshold_sigma` and the
quote proof need to convince a verifier who was not present.

`VoleScheme` is *designated verifier*. The prover holds `(x, M)` and the
verifier holds `(K, Delta)` with `M = K + Delta*x`; nothing is published, and
only the holder of `Delta` can check an opening. Making it publicly verifiable
is what VOLE-in-the-Head does (Baum et al., CRYPTO 2023), at about twice the
communication of the designated-verifier protocol --- and that transform is
**not implemented here**. What is implemented is the commitment underneath it,
so the cost of the homomorphic operations can be measured against Pedersen's.

Binding for the VOLE scheme is `1/|F|` per opening rather than computational,
and hiding is perfect given that `K` is uniform. Neither is a discrete-log
assumption, which is why the scheme is post-quantum and why there is no group
order for the MPC field to match.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .commit import Pedersen
from .groups import DOMAIN


@runtime_checkable
class CommitmentScheme(Protocol):
    """What a protocol in this package needs from a commitment scheme."""

    name: str
    publicly_verifiable: bool

    @property
    def scalar_modulus(self) -> int: ...

    def commit(self, value: int, blinding: int) -> Any: ...
    def random_blinding(self) -> int: ...
    def random_scalar(self) -> int: ...
    def add(self, a: Any, b: Any) -> Any: ...
    def scale(self, commitment: Any, scalar: int) -> Any: ...
    def negate(self, commitment: Any) -> Any: ...
    def zero(self) -> Any: ...
    def encode(self, commitment: Any) -> bytes: ...

    def equal(self, a: Any, b: Any) -> bool:
        return self.encode(a) == self.encode(b)


class PedersenScheme:
    """The scheme this repository already uses, behind the seam.

    Every operation is one or two scalar multiplications on the curve, which is
    where the cost is: `add` is a group multiplication and `scale` is a full
    scalar multiplication.
    """

    name = "pedersen"
    publicly_verifiable = True

    def __init__(self, key: Pedersen):
        self.key = key
        self.group = key.group

    @property
    def scalar_modulus(self) -> int:
        return self.group.order

    def commit(self, value: int, blinding: int):
        return self.key.commit(value, blinding)

    def random_blinding(self) -> int:
        return self.key.random_blinding()

    def random_scalar(self) -> int:
        return self.group.random_scalar()

    def add(self, a, b):
        return self.group.mul(a, b)

    def scale(self, commitment, scalar: int):
        return self.group.point_pow(commitment, scalar % self.group.order)

    def negate(self, commitment):
        return self.group.neg(commitment)

    def zero(self):
        return self.group.identity()

    def encode(self, commitment) -> bytes:
        return self.group.encode(commitment)

    def equal(self, a, b) -> bool:
        return self.encode(a) == self.encode(b)


@dataclass(frozen=True)
class VoleCommitment:
    """One MAC. `tag` is the prover's share; the verifier recomputes it.

    Carrying the value alongside the tag is what a prover holds anyway --- it
    has to open eventually --- and it keeps the homomorphic operations honest:
    combining commitments has to combine the values the same way, or the
    opening stops matching.
    """

    value: int
    tag: int


class VoleScheme:
    """A linearly homomorphic commitment from a VOLE correlation.

    `M = K + Delta*x` over a prime field. The prover holds `(x, M)`, the
    verifier holds `(K, Delta)`. Adding two commitments adds the values and the
    tags; scaling by a public constant scales both. Every operation is field
    arithmetic, which is the reason to measure it.

    This is the *designated-verifier* primitive. Section 7 of `BINDING.md` says
    what the public-verifiability transform costs and that it is not here.
    """

    name = "vole"
    publicly_verifiable = False

    # 2^127 - 1 is prime, so binding fails with probability about 2^-127 an
    # opening --- the same order as the curve, without the curve.
    DEFAULT_MODULUS = (1 << 127) - 1

    def __init__(self, modulus: int | None = None, delta: int | None = None,
                 label: bytes = b"qomm:vole:v1"):
        self.modulus = modulus or self.DEFAULT_MODULUS
        self.label = label
        # the verifier's secret. A prover that learned it could open to any
        # value, which is exactly the binding assumption.
        self._delta = delta if delta is not None else self.random_scalar()
        self._keys: dict[int, int] = {}
        self._next = 0

    @property
    def scalar_modulus(self) -> int:
        return self.modulus

    def random_scalar(self) -> int:
        return secrets.randbelow(self.modulus - 1) + 1

    def random_blinding(self) -> int:
        return self.random_scalar()

    def commit(self, value: int, blinding: int) -> VoleCommitment:
        """`blinding` is the verifier's key here, which is what makes it hide.

        The signature matches the Pedersen one so the protocols above do not
        have to know which scheme they are on. What differs is the meaning: the
        blinding is not a second exponent but the key the verifier holds.
        """
        key = blinding % self.modulus
        tag = (key + self._delta * value) % self.modulus
        return VoleCommitment(value % self.modulus, tag)

    def add(self, a: VoleCommitment, b: VoleCommitment) -> VoleCommitment:
        return VoleCommitment((a.value + b.value) % self.modulus,
                              (a.tag + b.tag) % self.modulus)

    def scale(self, commitment: VoleCommitment, scalar: int) -> VoleCommitment:
        k = scalar % self.modulus
        return VoleCommitment((commitment.value * k) % self.modulus,
                              (commitment.tag * k) % self.modulus)

    def negate(self, commitment: VoleCommitment) -> VoleCommitment:
        return VoleCommitment((-commitment.value) % self.modulus,
                              (-commitment.tag) % self.modulus)

    def zero(self) -> VoleCommitment:
        return VoleCommitment(0, 0)

    def encode(self, commitment: VoleCommitment) -> bytes:
        width = (self.modulus.bit_length() + 7) // 8
        return (commitment.value.to_bytes(width, "big")
                + commitment.tag.to_bytes(width, "big"))

    def equal(self, a: VoleCommitment, b: VoleCommitment) -> bool:
        return a.value == b.value and a.tag == b.tag

    # --- what only this scheme has -----------------------------------------

    def opens(self, commitment: VoleCommitment, value: int, key: int) -> bool:
        """The verifier's check, which needs Delta and so is not public."""
        return commitment.tag == (key + self._delta * value) % self.modulus

    def commitment_bytes(self) -> int:
        """What one commitment costs on the wire, for the comparison."""
        return 2 * ((self.modulus.bit_length() + 7) // 8)


def make_scheme(name: str, **kwargs) -> CommitmentScheme:
    """One place that knows the names, so a runner can take a flag."""
    if name == "pedersen":
        from .groups import make_group
        group = kwargs.pop("group", "ed25519")
        label = kwargs.pop("label", b"qomm:pedersen:v1")
        return PedersenScheme(Pedersen(make_group(group), label))
    if name == "vole":
        return VoleScheme(**kwargs)
    raise ValueError(f"unknown commitment scheme {name}; "
                     f"choose from ['pedersen', 'vole']")
