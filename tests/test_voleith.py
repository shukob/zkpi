"""The VOLE-in-the-Head transform, checked at every seam it has.

The tree, the correlation, the proof, and every way of tampering with what is
published. The correlation test is the one that matters: it checks the identity
the whole construction rests on, `Q = Delta*u + V`, directly rather than through
a proof that could pass for the wrong reason.
"""

from __future__ import annotations

import secrets

import pytest

from zk import voleith as V
from zk.scheme import (LinearProofScheme, PedersenLinearProof,
                       VoleInTheHeadLinearProof, make_linear_proof)

P = (1 << 127) - 1


def values(n: int, mask_bits: int = 119) -> list[int]:
    return [secrets.randbelow(1 << 31) for _ in range(n - 1)] + [
        secrets.randbelow(1 << mask_bits)]


# --- the tree -------------------------------------------------------------

@pytest.mark.parametrize("depth", [1, 2, 3, 5, 8])
def test_copath_opens_every_leaf_but_one(depth):
    root = secrets.token_bytes(V.SEED_BYTES)
    leaves = V.expand_tree(root, depth, rep=0)
    assert len(leaves) == 1 << depth
    for index in range(1 << depth):
        opened = V.open_copath(V.copath(root, depth, 0, index), depth, 0, index)
        assert opened[index] is None, "the punctured leaf must stay hidden"
        for other in range(1 << depth):
            if other != index:
                assert opened[other] == leaves[other], (index, other)


def test_copath_is_log_sized():
    root = secrets.token_bytes(V.SEED_BYTES)
    assert len(V.copath(root, 10, 0, 512)) == 10       # not 1024


def test_copath_rejects_an_index_outside_the_tree():
    root = secrets.token_bytes(V.SEED_BYTES)
    with pytest.raises(ValueError):
        V.copath(root, 4, 0, 16)


def test_trees_at_different_repetitions_differ():
    root = secrets.token_bytes(V.SEED_BYTES)
    assert V.expand_tree(root, 4, rep=0) != V.expand_tree(root, 4, rep=1)


def test_a_leaf_commitment_binds_its_index_and_repetition():
    leaf = secrets.token_bytes(V.SEED_BYTES)
    seen = {V.leaf_commitment(leaf, r, i) for r in range(3) for i in range(3)}
    assert len(seen) == 9


# --- the correlation the whole thing rests on -----------------------------

@pytest.mark.parametrize("depth", [3, 6])
def test_the_vole_correlation_holds(depth):
    """`Q = Delta*u + V` from what each side actually holds."""
    n, rep = 4, 0
    root = secrets.token_bytes(V.SEED_BYTES)
    pack = V.Packing(P, n, depth)
    mask = pack.mask
    leaves = V.expand_tree(root, depth, rep)

    u_packed = w_packed = 0
    for index, leaf in enumerate(leaves):
        t = pack.leaf(leaf, rep, index, mask)
        u_packed += t
        w_packed += index * t
    u = pack.unpack(u_packed)
    v = [(-x) % P for x in pack.unpack(w_packed)]

    delta = secrets.randbelow(1 << depth)
    opened = V.open_copath(V.copath(root, depth, rep, delta), depth, rep, delta)
    s_packed = x_packed = 0
    for index, leaf in enumerate(opened):
        if index == delta:
            continue
        t = pack.leaf(leaf, rep, index, mask)
        s_packed += t
        x_packed += index * t
    totals, weights = pack.unpack(s_packed), pack.unpack(x_packed)

    for i in range(n):
        q = (delta * totals[i] - weights[i]) % P
        assert q == (delta * u[i] + v[i]) % P, i


def test_packing_leaves_room_for_every_sum():
    """No slot can carry into the next one, at any parameter this module uses."""
    for depth in (4, 8, 12, 16):
        pack = V.Packing(P, 200, depth)
        widest = ((1 << pack.value_bits) - 1) * (1 << depth) * (1 << depth)
        assert widest < (1 << (pack.slot_bytes * 8)), depth


def test_leaf_values_reach_past_the_modulus():
    """`d = w - u` hides `w` only if `u` is close to uniform mod p."""
    pack = V.Packing(P, 4, 8)
    assert pack.value_bits >= P.bit_length() + 64


# --- the proof ------------------------------------------------------------

def make(n=12, depth=5, repeats=4, context=b"ctx", vals=None):
    vals = vals if vals is not None else values(n)
    prover = V.Prover(P, depth=depth, repeats=repeats)
    root, correction, _ = prover.commit(vals)
    coeffs = V.coefficients(root, correction, context, P, 40, len(vals))
    return vals, coeffs, prover.prove(coeffs, context)


def test_an_honest_proof_verifies():
    _, coeffs, proof = make()
    assert V.verify(proof, coeffs, b"ctx") == (True, "ok")


def test_the_opening_is_the_combination():
    vals, coeffs, proof = make()
    assert proof.opening == sum(c * v for c, v in zip(coeffs, vals)) % P


@pytest.mark.parametrize("depth,repeats", [(2, 3), (4, 4), (6, 2), (8, 2)])
def test_it_verifies_at_every_shape(depth, repeats):
    _, coeffs, proof = make(n=7, depth=depth, repeats=repeats)
    assert V.verify(proof, coeffs, b"ctx")[0]
    assert proof.soundness_bits() == depth * repeats


def test_one_value_is_allowed_and_none_is_not():
    assert make(n=1)[2].n_values == 1
    with pytest.raises(ValueError):
        V.Prover(P, 4, 2).commit([])


def test_a_degenerate_shape_is_refused():
    with pytest.raises(ValueError):
        V.Prover(P, depth=0, repeats=4)
    with pytest.raises(ValueError):
        V.Prover(P, depth=4, repeats=0)


# --- tampering ------------------------------------------------------------

def swap(proof, **kw):
    fields = dict(root=proof.root, witness_correction=proof.witness_correction,
                  vole_corrections=proof.vole_corrections, opening=proof.opening,
                  tags=proof.tags, copaths=proof.copaths,
                  punctured=proof.punctured, depth=proof.depth,
                  repeats=proof.repeats, modulus=proof.modulus)
    fields.update(kw)
    return V.LinearProof(**fields)


def test_a_changed_opening_is_caught():
    _, coeffs, proof = make()
    assert not V.verify(swap(proof, opening=(proof.opening + 1) % P), coeffs, b"ctx")[0]


def test_a_changed_witness_correction_is_caught():
    _, coeffs, proof = make()
    bad = list(proof.witness_correction)
    bad[0] = (bad[0] + 1) % P
    assert not V.verify(swap(proof, witness_correction=bad), coeffs, b"ctx")[0]


def test_a_changed_vole_correction_is_caught():
    _, coeffs, proof = make()
    bad = [list(row) for row in proof.vole_corrections]
    bad[0][0] = (bad[0][0] + 1) % P
    assert not V.verify(swap(proof, vole_corrections=bad), coeffs, b"ctx")[0]


def test_a_changed_tag_is_caught():
    _, coeffs, proof = make()
    bad = list(proof.tags)
    bad[0] = (bad[0] + 1) % P
    assert not V.verify(swap(proof, tags=bad), coeffs, b"ctx")[0]


def test_a_swapped_copath_seed_is_caught():
    _, coeffs, proof = make()
    bad = [list(p) for p in proof.copaths]
    bad[0][0] = secrets.token_bytes(V.SEED_BYTES)
    ok, why = V.verify(swap(proof, copaths=bad), coeffs, b"ctx")
    assert not ok and ("not the committed ones" in why or "does not hold" in why)


def test_a_swapped_punctured_commitment_is_caught():
    _, coeffs, proof = make()
    bad = list(proof.punctured)
    bad[0] = secrets.token_bytes(V.COMMIT_BYTES)
    ok, why = V.verify(swap(proof, punctured=bad), coeffs, b"ctx")
    assert not ok and "not the committed ones" in why


def test_a_changed_root_is_caught():
    _, coeffs, proof = make()
    assert not V.verify(swap(proof, root=secrets.token_bytes(V.COMMIT_BYTES)),
                        coeffs, b"ctx")[0]


def test_another_context_does_not_verify():
    _, coeffs, proof = make()
    assert not V.verify(proof, coeffs, b"a different auction")[0]


def test_changed_coefficients_do_not_verify():
    _, coeffs, proof = make()
    bad = list(coeffs)
    bad[0] += 1
    assert not V.verify(proof, bad, b"ctx")[0]


def test_a_substituted_value_cannot_be_proved():
    """The point of the whole thing: the opening has to be of the committed vector."""
    vals = values(9)
    prover = V.Prover(P, depth=5, repeats=4)
    root, correction, _ = prover.commit(vals)
    coeffs = V.coefficients(root, correction, b"ctx", P, 40, len(vals))
    proof = prover.prove(coeffs, b"ctx")
    substituted = list(vals)
    substituted[0] += 1
    forged = swap(proof, opening=sum(c * v for c, v in zip(coeffs, substituted)) % P)
    assert not V.verify(forged, coeffs, b"ctx")[0]


def test_a_malformed_shape_is_rejected_rather_than_crashing():
    _, coeffs, proof = make()
    assert not V.verify(swap(proof, vole_corrections=[]), coeffs, b"ctx")[0]
    assert not V.verify(swap(proof, copaths=proof.copaths[:1]), coeffs, b"ctx")[0]
    assert not V.verify(proof, coeffs[:-1], b"ctx")[0]


# --- the one-time property ------------------------------------------------

def test_a_second_proof_is_refused():
    """eprint 2026/337: these commitments are one-time. Refuse, do not document."""
    prover = V.Prover(P, depth=4, repeats=3)
    root, correction, _ = prover.commit(values(5))
    coeffs = V.coefficients(root, correction, b"ctx", P, 40, 5)
    prover.prove(coeffs, b"ctx")
    with pytest.raises(V.OneTimeError):
        prover.prove(coeffs, b"another statement")


def test_a_second_commitment_is_refused():
    prover = V.Prover(P, depth=4, repeats=3)
    prover.commit(values(5))
    with pytest.raises(V.OneTimeError):
        prover.commit(values(5))


def test_proving_before_committing_is_refused():
    with pytest.raises(RuntimeError):
        V.Prover(P, 4, 3).prove([1, 2, 3], b"ctx")


# --- the coefficients -----------------------------------------------------

def test_coefficients_depend_on_the_commitment():
    root = secrets.token_bytes(V.COMMIT_BYTES)
    a = V.coefficients(root, [1, 2, 3], b"ctx", P, 40, 3)
    b = V.coefficients(root, [1, 2, 4], b"ctx", P, 40, 3)
    assert a != b


def test_coefficients_depend_on_the_context():
    root = secrets.token_bytes(V.COMMIT_BYTES)
    assert (V.coefficients(root, [1, 2, 3], b"a", P, 40, 3)
            != V.coefficients(root, [1, 2, 3], b"b", P, 40, 3))


def test_no_coefficient_is_zero():
    """A zero coefficient would leave that value unchecked."""
    root = secrets.token_bytes(V.COMMIT_BYTES)
    assert all(c > 0 for c in V.coefficients(root, list(range(64)), b"ctx", P, 8, 64))


def test_the_challenge_covers_everything_the_prover_sends():
    # a wide depth on purpose: at the default depth 8 the challenge is two
    # bytes and this test collides once in 65,536 runs, which it duly did
    args = dict(root=secrets.token_bytes(32), correction=[1, 2],
                vole_corrections=[[3, 4]], opening=5, tags=[6, 7],
                context=b"ctx", modulus=P, depth=64, repeats=4)
    base = V.challenge(**args)
    for key, other in [("correction", [1, 3]), ("vole_corrections", [[3, 5]]),
                       ("opening", 6), ("tags", [6, 8]), ("context", b"other"),
                       ("root", secrets.token_bytes(32))]:
        assert V.challenge(**{**args, key: other}) != base, key


# --- size -----------------------------------------------------------------

def test_the_size_arithmetic_matches_a_real_proof():
    _, _, proof = make(n=12, depth=5, repeats=4)
    predicted = V.proof_size(12, P.bit_length(), 5, 4)
    del predicted["total"], predicted["hashes"], predicted["soundness_bits"]
    assert proof.size_breakdown() == predicted


def test_the_corrections_are_what_dominates_over_a_prime_field():
    """The finding: `(repeats-1)*ell` field elements, not the tree openings."""
    parts = V.proof_size(167, 127, 8, 16)
    assert parts["vole_corrections"] > 0.8 * parts["total"]
    assert parts["copaths"] < 0.1 * parts["total"]


def test_a_deeper_tree_trades_size_for_hashing():
    """FAEST table 2 in miniature: bigger q, smaller proof, more computation."""
    shallow = V.proof_size(167, 127, 6, 22)
    deep = V.proof_size(167, 127, 12, 11)
    assert deep["total"] < shallow["total"]
    assert deep["hashes"] > shallow["hashes"]
    assert min(shallow["soundness_bits"], deep["soundness_bits"]) >= 128


# --- the seam -------------------------------------------------------------

@pytest.mark.parametrize("name", ["pedersen", "voleith"])
def test_both_schemes_satisfy_the_seam(name):
    scheme = make_linear_proof(name)
    assert isinstance(scheme, LinearProofScheme)
    assert scheme.publicly_verifiable, "this seam is only for public verification"
    vals = values(10)
    proof = scheme.prove_linear(vals, b"ctx")
    assert scheme.verify_linear(proof, b"ctx") == (True, "ok")
    assert scheme.proof_bytes(proof) == sum(scheme.size_breakdown(proof).values())


@pytest.mark.parametrize("name", ["pedersen", "voleith"])
def test_a_proof_does_not_verify_under_another_context(name):
    scheme = make_linear_proof(name)
    proof = scheme.prove_linear(values(10), b"ctx")
    assert not scheme.verify_linear(proof, b"elsewhere")[0]


def test_only_one_of_the_two_is_post_quantum():
    assert not PedersenLinearProof.post_quantum
    assert VoleInTheHeadLinearProof.post_quantum


def test_an_unknown_scheme_names_the_ones_that_exist():
    with pytest.raises(ValueError, match="pedersen"):
        make_linear_proof("groth16")


# --- what a general linear code would and would not buy ---------------------

def test_the_code_swap_alone_still_reproduces_the_figure_it_used_to_report():
    """10,816 B at [31,16,16] --- kept so the correction is a visible delta.

    This branch counts the corrections a general code replaces and nothing
    else. It is the number this project reported for a while, and pinning it
    means the next reader can see exactly which term was missing rather than
    having to trust that something changed.
    """
    from scripts.run_voleith import linear_code_arithmetic

    best = linear_code_arithmetic(167, depth=8)["code_swap_only"]
    assert (best["k_C"], best["n_C"], best["d_C"]) == (16, 31, 16)
    assert best["bytes"] == 10816
    assert best["hashes"] == 15872


def test_the_protocol_that_makes_the_code_usable_costs_most_of_the_saving():
    """Pi_2D-LC, not a code swap. The paper's own '2x overhead', priced.

    The general code is homomorphic across the k_C-blocks and not within one,
    and our statement is one inner product with a distinct coefficient per
    value --- within a block. Recovering that needs figure 6, which calls the
    subspace VOLE for 2l+2 rows instead of l and opens an (l+1) x n_C matrix
    on top. The saving over the measured repetition-code proof survives, but
    it is about 2.4x rather than the 4.2x the code swap alone suggested.
    """
    from scripts.run_voleith import linear_code_arithmetic

    arithmetic = linear_code_arithmetic(167, depth=8)
    best = arithmetic["protocol_complete"]
    assert best["bytes"] == 18896 and best["k_C"] == 32
    # measured repetition-code proof and measured Pedersen proof, same statement
    assert round(45616 / best["bytes"], 1) == 2.4
    assert round(best["bytes"] / 5440, 1) == 3.5
    # and it is strictly worse than the branch that ignores the protocol
    assert best["bytes"] > arithmetic["code_swap_only"]["bytes"]
