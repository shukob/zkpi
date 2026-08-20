"""Soundness and zero-knowledge checks for the optimised OR-DLEQ proof.

A faster proof that accepts forgeries is worthless, so every optimisation
backend is put through the same negative controls. The backends must agree on
what they accept and reject, not merely on how fast they run.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from zk.groups import BACKENDS, make_group                                  # noqa: E402
from zk.or_dleq import (                                                    # noqa: E402
    OrDleqProver, OrDleqVerifier, Proof, Statement, build_registry,
)

ALL_BACKENDS = sorted(BACKENDS)
FAST_BACKENDS = ["modp_multiexp", "ed25519"]      # for the slower parametrised cases


def _setup(backend: str, size: int = 4, index: int = 1):
    group = make_group(backend)
    statement, secret_scalars = build_registry(group, size, seed=11)
    prover = OrDleqProver(group)
    verifier = OrDleqVerifier(group)
    proof = prover.prove(statement, secret_scalars[index], index)
    return group, statement, secret_scalars, prover, verifier, proof


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_honest_proof_verifies(backend):
    _, statement, _, _, verifier, proof = _setup(backend)
    assert verifier.verify(statement, proof)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_every_registry_position_can_prove(backend):
    group = make_group(backend)
    statement, secret_scalars = build_registry(group, 4, seed=11)
    prover, verifier = OrDleqProver(group), OrDleqVerifier(group)
    for index, secret in enumerate(secret_scalars):
        assert verifier.verify(statement, prover.prove(statement, secret, index))


@pytest.mark.parametrize("backend", FAST_BACKENDS)
def test_tampered_challenge_is_rejected(backend):
    group, statement, _, _, verifier, proof = _setup(backend)
    broken = list(proof.challenges)
    broken[0] = (broken[0] + 1) % group.order
    assert not verifier.verify(statement, Proof(proof.nullifier, tuple(broken), proof.responses))


@pytest.mark.parametrize("backend", FAST_BACKENDS)
def test_tampered_response_is_rejected(backend):
    group, statement, _, _, verifier, proof = _setup(backend)
    broken = list(proof.responses)
    broken[2] = (broken[2] + 1) % group.order
    assert not verifier.verify(statement, Proof(proof.nullifier, proof.challenges, tuple(broken)))


@pytest.mark.parametrize("backend", FAST_BACKENDS)
def test_swapped_nullifier_is_rejected(backend):
    group, statement, secret_scalars, prover, verifier, proof = _setup(backend)
    other = group.point_pow(group.hash_to_point(statement.scope.encode()), secret_scalars[0])
    assert not verifier.verify(statement, Proof(other, proof.challenges, proof.responses))


@pytest.mark.parametrize("backend", FAST_BACKENDS)
def test_proof_does_not_transfer_to_another_scope(backend):
    _, statement, _, _, verifier, proof = _setup(backend)
    elsewhere = Statement(statement.registry_id, statement.points,
                          "qomm:other-scope", statement.context_hash)
    assert not verifier.verify(elsewhere, proof)


@pytest.mark.parametrize("backend", FAST_BACKENDS)
def test_proof_is_bound_to_its_context(backend):
    _, statement, _, _, verifier, proof = _setup(backend)
    rebound = Statement(statement.registry_id, statement.points, statement.scope,
                        hashlib.sha256(b"different context").hexdigest())
    assert not verifier.verify(rebound, proof)


@pytest.mark.parametrize("backend", FAST_BACKENDS)
def test_outsider_cannot_prove(backend):
    group = make_group(backend)
    statement, _ = build_registry(group, 4, seed=11)
    outsider = group.random_scalar()
    prover, verifier = OrDleqProver(group), OrDleqVerifier(group)
    # the prover happily produces something; it must not verify
    forged = prover.prove(statement, outsider, 0)
    assert not verifier.verify(statement, forged)


@pytest.mark.parametrize("backend", FAST_BACKENDS)
def test_out_of_range_scalars_are_rejected(backend):
    group, statement, _, _, verifier, proof = _setup(backend)
    bad = list(proof.responses)
    bad[0] = group.order            # exactly out of range
    assert not verifier.verify(statement, Proof(proof.nullifier, proof.challenges, tuple(bad)))


@pytest.mark.parametrize("backend", FAST_BACKENDS)
def test_nullifier_links_within_a_scope_and_not_across(backend):
    group = make_group(backend)
    statement, secret_scalars = build_registry(group, 4, seed=11)
    prover = OrDleqProver(group)
    first = prover.prove(statement, secret_scalars[1], 1)
    again = prover.prove(statement, secret_scalars[1], 1)
    other_scope = Statement(statement.registry_id, statement.points,
                            "qomm:another", statement.context_hash)
    elsewhere = prover.prove(other_scope, secret_scalars[1], 1)
    neighbour = prover.prove(statement, secret_scalars[2], 2)
    assert group.encode(first.nullifier) == group.encode(again.nullifier)
    assert group.encode(first.nullifier) != group.encode(elsewhere.nullifier)
    assert group.encode(first.nullifier) != group.encode(neighbour.nullifier)


@pytest.mark.parametrize("backend", FAST_BACKENDS)
def test_two_proofs_of_the_same_statement_differ(backend):
    """Fresh randomness each time, so proofs are not a fingerprint of the holder."""
    group = make_group(backend)
    statement, secret_scalars = build_registry(group, 4, seed=11)
    prover = OrDleqProver(group)
    first = prover.prove(statement, secret_scalars[1], 1)
    second = prover.prove(statement, secret_scalars[1], 1)
    assert first.challenges != second.challenges
    assert first.responses != second.responses


def test_backends_agree_on_the_group_arithmetic():
    """The MODP backends differ only in how they compute the same value."""
    reference = make_group("modp_naive")
    base = reference.base_pow(3)
    point = reference.base_pow(5)
    expected = reference.commit(base, 7, point, 11)
    for name in ("modp_inv", "modp_negexp", "modp_multiexp"):
        assert make_group(name).commit(base, 7, point, 11) == expected


def test_ed25519_scope_generator_has_no_known_discrete_log():
    """A generator derived as t*G would make the nullifier publicly computable."""
    group = make_group("ed25519")
    scope_point = group.hash_to_point(b"qomm:quote:v1")
    assert group.is_valid(scope_point)
    # a cheap sanity bound: it must not be a small multiple of the base point
    for candidate in range(1, 2000):
        assert group.base_pow(candidate) != scope_point


def test_ed25519_rejects_points_off_the_prime_order_subgroup():
    group = make_group("ed25519")
    assert not group.is_valid(b"\x00" * 32)
    assert not group.is_valid(b"\x01" + b"\x00" * 31)   # identity encoding
    assert not group.is_valid(b"\xff" * 32)
    assert not group.is_valid(b"short")


def test_hash_to_point_lands_in_the_prime_order_subgroup():
    """Regression: libsodium's point validity check is not a subgroup check.

    Measured on this build, about 39% of the encodings that
    ``crypto_core_ed25519_is_valid_point`` accepts are not of order L. Using one
    of those as a generator leaves every proof syntactically valid while
    silently breaking the prime-order assumption the soundness argument needs,
    so the generator has to have its cofactor cleared.
    """
    group = make_group("ed25519")
    for label in (b"qomm:pedersen:v1", b"qomm:quote:v1", b"scope-a", b"scope-b"):
        point = group.hash_to_point(label)
        assert group.is_valid(point)
        assert group.encode(group.mul(group.point_pow(point, group.order - 1), point)) \
            == group.encode(group.identity())
        assert group.encode(group.point_pow(point, group.order - 1)) == group.encode(group.neg(point))


@pytest.mark.parametrize("backend", ["modp_negexp", "ed25519"])
def test_group_negation_is_a_true_inverse(backend):
    group = make_group(backend)
    point = group.mul(group.base_pow(3), group.point_pow(group.hash_to_point(b"x"), 7))
    assert group.encode(group.mul(point, group.neg(point))) == group.encode(group.identity())
    assert group.encode(group.point_pow(point, group.order - 1)) == group.encode(group.neg(point))


# --- the alternative we previously argued away without measuring -----------

def test_groth_kohlweiss_proves_and_verifies():
    from zk.commit import Pedersen
    from zk.gk_oneofmany import GkProver, GkVerifier, build_set

    group = make_group("ed25519")
    key = Pedersen(group, b"qomm:gk:v1")
    prover, verifier = GkProver(group, key), GkVerifier(group, key)
    for size in (4, 8, 16):
        for index in (0, size // 2, size - 1):
            commitments, randomness = build_set(group, size, index, key)
            proof = prover.prove(commitments, index, randomness)
            assert verifier.verify(commitments, proof), f"size={size} index={index}"


def test_groth_kohlweiss_rejects_a_set_without_the_witness():
    from zk.commit import Pedersen
    from zk.gk_oneofmany import GkProver, GkVerifier, build_set

    group = make_group("ed25519")
    key = Pedersen(group, b"qomm:gk:v1")
    commitments, randomness = build_set(group, 8, 3, key)
    proof = GkProver(group, key).prove(commitments, 3, randomness)
    other, _ = build_set(group, 8, 5, key)
    assert not GkVerifier(group, key).verify(other, proof)


def test_groth_kohlweiss_proof_is_logarithmic_and_or_proof_is_linear():
    """The size crossover is the reason to keep both implementations."""
    from zk.commit import Pedersen
    from zk.gk_oneofmany import GkProver, build_set
    from zk.or_dleq import OrDleqProver, build_registry

    group = make_group("ed25519")
    key = Pedersen(group, b"qomm:gk:v1")
    sizes = (8, 64)
    gk_sizes, or_sizes = [], []
    for size in sizes:
        commitments, randomness = build_set(group, size, 0, key)
        gk_sizes.append(GkProver(group, key).prove(commitments, 0, randomness).size_bytes(group))
        statement, secrets = build_registry(group, size, seed=7)
        or_sizes.append(OrDleqProver(group).prove(statement, secrets[0], 0).size_bytes(group))
    # eight-fold set growth: the OR proof grows eight-fold, GK by a constant per doubling
    assert or_sizes[1] / or_sizes[0] > 7
    assert gk_sizes[1] / gk_sizes[0] < 2.5
    # and at the larger size GK is the smaller proof
    assert gk_sizes[1] < or_sizes[1]
