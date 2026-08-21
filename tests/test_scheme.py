"""The commitment layer is a choice, and the two choices differ in what they promise."""

from __future__ import annotations

import pytest

from zk.scheme import CommitmentScheme, PedersenScheme, VoleScheme, make_scheme

NAMES = ("pedersen", "vole")


@pytest.mark.parametrize("name", NAMES)
def test_it_satisfies_the_protocol(name: str) -> None:
    assert isinstance(make_scheme(name), CommitmentScheme)


@pytest.mark.parametrize("name", NAMES)
def test_the_homomorphism_holds(name: str) -> None:
    """`3a + b` has to commit to `3*value(a) + value(b)`, whatever the scheme."""
    scheme = make_scheme(name)
    ra, rb = scheme.random_blinding(), scheme.random_blinding()
    a, b = scheme.commit(5, ra), scheme.commit(8, rb)
    combined = scheme.add(scheme.scale(a, 3), b)
    direct = scheme.commit(3 * 5 + 8, (3 * ra + rb) % scheme.scalar_modulus)
    assert scheme.equal(combined, direct)


@pytest.mark.parametrize("name", NAMES)
def test_negate_and_zero_are_consistent(name: str) -> None:
    scheme = make_scheme(name)
    r = scheme.random_blinding()
    c = scheme.commit(41, r)
    assert scheme.equal(scheme.add(c, scheme.negate(c)), scheme.zero())


@pytest.mark.parametrize("name", NAMES)
def test_different_values_commit_differently(name: str) -> None:
    scheme = make_scheme(name)
    r = scheme.random_blinding()
    assert not scheme.equal(scheme.commit(1, r), scheme.commit(2, r))


def test_only_one_of_them_is_publicly_verifiable() -> None:
    """The difference the seam must not hide, because the quote proof needs it."""
    assert make_scheme("pedersen").publicly_verifiable is True
    assert make_scheme("vole").publicly_verifiable is False


def test_an_unknown_scheme_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="unknown commitment scheme"):
        make_scheme("lattice")


def test_the_vole_binding_needs_delta() -> None:
    """A prover that knew Delta could open to anything, which is the assumption."""
    scheme = VoleScheme()
    key = scheme.random_blinding()
    c = scheme.commit(999, key)
    assert scheme.opens(c, 999, key)
    assert not scheme.opens(c, 1000, key)
    other = VoleScheme(modulus=scheme.scalar_modulus)
    assert not other.opens(c, 999, key), "a different Delta accepted the opening"
