"""Negative controls for the market-maker policy audit and the KYB credential.

The audit is only worth running if it rejects a policy that breaks the venue's
rules, and the credential is only worth issuing if it cannot be replayed,
re-cohorted or multiplied across wallets. Those are the cases here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from zk.commit import Pedersen, prove_bit                                    # noqa: E402
from zk.groups import make_group                                             # noqa: E402
from zk.kyb import (                                                         # noqa: E402
    BusinessAttributes, EntityLimits, EntityRateLimiter, KybIssuer,
    SignedCohortRegistry, cohort_id, present, scope_nullifier,
    verify_presentation, verify_registry,
)
from zk.policy_audit import (                                                # noqa: E402
    PolicyAuditor, PolicyBounds, PolicyCommitter, PolicyShare, reconstruct,
)

GOOD_POLICY = dict(mid=100_000, half=14, slope=2, invcoef=1, inv=-320,
                   maxqty=400, expiry=1_600, active=1)
REF_MID, NOW, HORIZON = 100_000, 1_000, 3_600


@pytest.fixture(scope="module")
def group():
    return make_group("ed25519")


def _audit(group, policy=None, **kw):
    committer = PolicyCommitter(group)
    nullifier = group.hash_to_point(b"entity-nullifier")
    audit, shares, blindings = committer.audit(
        dict(policy or GOOD_POLICY), ref_mid=REF_MID, now_t=NOW,
        n_parties=7, threshold=2, entity_nullifier=nullifier, **kw)
    return committer, audit, shares, blindings


# --- policy audit ---------------------------------------------------------

def test_well_formed_policy_is_accepted(group):
    _, audit, _, _ = _audit(group)
    ok, reason = PolicyAuditor(group).verify(
        audit, now_t=NOW, ref_mid=REF_MID, max_horizon=HORIZON)
    assert ok, reason


def test_every_node_can_check_its_own_share(group):
    committer, audit, shares, _ = _audit(group)
    for name, field_shares in shares.items():
        assert len(field_shares) == 7
        for share in field_shares:
            assert committer.verify_share(share, audit.fields[name])


def test_a_tampered_share_is_caught_by_its_node(group):
    committer, audit, shares, _ = _audit(group)
    victim = shares["maxqty"][3]
    forged = PolicyShare(victim.party, victim.value_share + 1, victim.blinding_share)
    assert not committer.verify_share(forged, audit.fields["maxqty"])


def test_shares_reconstruct_the_committed_value(group):
    group_order = group.order
    committer, audit, shares, _ = _audit(group)
    assert reconstruct(group, shares["maxqty"], 2) == GOOD_POLICY["maxqty"]
    assert reconstruct(group, shares["inv"], 2) == GOOD_POLICY["inv"] % group_order
    # a threshold-sized subset is enough, and any subset agrees
    subset = [shares["half"][i] for i in (1, 4, 6)]
    assert reconstruct(group, subset, 2) == GOOD_POLICY["half"]


@pytest.mark.parametrize("field,value", [
    ("half", 900),        # spread far outside the venue band
    ("slope", 100),
    ("invcoef", 50),
    ("maxqty", 5_000),    # more size than the collateral supports
    ("inv", 50_000),
    ("mid", 400_000),     # not anchored to the reference price
])
def test_out_of_band_fields_cannot_be_proved(group, field, value):
    policy = dict(GOOD_POLICY)
    policy[field] = value
    with pytest.raises(ValueError):
        _audit(group, policy)


def test_swapping_in_another_commitment_is_rejected(group):
    """A maker cannot keep a valid proof and quote a different hidden spread."""
    committer, audit, _, _ = _audit(group)
    key = Pedersen(group, b"qomm:policy:v1")
    forged_commitment = key.commit(900, key.random_blinding())
    from zk.policy_audit import FieldCommitment
    tampered = dict(audit.fields)
    tampered["half"] = FieldCommitment(
        forged_commitment, (forged_commitment,) + audit.fields["half"].coefficient_commitments[1:])
    broken = type(audit)(**{**audit.__dict__, "fields": tampered})
    ok, reason = PolicyAuditor(group).verify(
        broken, now_t=NOW, ref_mid=REF_MID, max_horizon=HORIZON)
    assert not ok and "half" in reason


def test_commitment_must_match_the_sharing(group):
    committer, audit, _, _ = _audit(group)
    from zk.policy_audit import FieldCommitment
    key = Pedersen(group, b"qomm:policy:v1")
    other = key.commit(7, key.random_blinding())
    tampered = dict(audit.fields)
    tampered["slope"] = FieldCommitment(audit.fields["slope"].commitment,
                                        (other,) + audit.fields["slope"].coefficient_commitments[1:])
    broken = type(audit)(**{**audit.__dict__, "fields": tampered})
    ok, reason = PolicyAuditor(group).verify(
        broken, now_t=NOW, ref_mid=REF_MID, max_horizon=HORIZON)
    assert not ok and "sharing" in reason


def test_audit_is_bound_to_the_reference_state(group):
    _, audit, _, _ = _audit(group)
    auditor = PolicyAuditor(group)
    assert not auditor.verify(audit, now_t=NOW + 1, ref_mid=REF_MID, max_horizon=HORIZON)[0]
    assert not auditor.verify(audit, now_t=NOW, ref_mid=REF_MID + 1, max_horizon=HORIZON)[0]


def test_expiry_must_sit_inside_the_horizon(group):
    _, audit, _, _ = _audit(group)
    auditor = PolicyAuditor(group)
    assert not auditor.verify(audit, now_t=NOW, ref_mid=REF_MID, max_horizon=10)[0]
    stale = type(audit)(**{**audit.__dict__, "expiry": NOW - 1})
    assert not auditor.verify(stale, now_t=NOW, ref_mid=REF_MID, max_horizon=HORIZON)[0]


def test_active_flag_must_be_a_bit(group):
    committer, audit, _, blindings = _audit(group)
    key = Pedersen(group, b"qomm:policy:v1")
    blinding = key.random_blinding()
    two = key.commit(2, blinding)
    context = committer._context(REF_MID, NOW, GOOD_POLICY["expiry"], audit.entity_nullifier)
    forged = prove_bit(key, two, 1, blinding, context + b":active")
    broken = type(audit)(**{**audit.__dict__, "active_commitment": two, "active_proof": forged})
    ok, reason = PolicyAuditor(group).verify(
        broken, now_t=NOW, ref_mid=REF_MID, max_horizon=HORIZON)
    assert not ok and "bit" in reason


def test_policy_can_be_attributed_to_its_entity(group):
    issuer_key = Ed25519PrivateKey.generate()
    _, audit, _, _ = _audit(group, entity_signer=issuer_key.sign)
    public = issuer_key.public_key()

    def check(digest: bytes, signature: bytes) -> bool:
        try:
            public.verify(signature, digest)
            return True
        except Exception:
            return False

    ok, reason = PolicyAuditor(group).verify(
        audit, now_t=NOW, ref_mid=REF_MID, max_horizon=HORIZON, entity_verifier=check)
    assert ok, reason
    unsigned = type(audit)(**{**audit.__dict__, "entity_signature": b"\x00" * 64})
    assert not PolicyAuditor(group).verify(
        unsigned, now_t=NOW, ref_mid=REF_MID, max_horizon=HORIZON, entity_verifier=check)[0]


# --- KYB ------------------------------------------------------------------

@pytest.fixture()
def venue(group):
    issuer = KybIssuer(group)
    entities = {
        "GROUP-A": issuer.enroll("GROUP-A", BusinessAttributes("JP", "bank", 4)),
        "GROUP-B": issuer.enroll("GROUP-B", BusinessAttributes("JP", "bank", 2)),
        "GROUP-C": issuer.enroll("GROUP-C", BusinessAttributes("JP", "bank", 5)),
    }
    registry = issuer.publish(cohort_id("JP", "bank", 2), registry_epoch=1, expires_at=9_999)
    return issuer, entities, registry


CONTEXT = {"venue": "qomm", "asset": 1}
SCOPE = "qomm:quote:epoch7"


def test_qualified_entity_presents_anonymously(group, venue):
    issuer, entities, registry = venue
    presentation = present(group, entities["GROUP-A"], registry, scope=SCOPE, context=CONTEXT)
    ok, reason = verify_presentation(group, presentation, registry, issuer.public_key,
                                     scope=SCOPE, context=CONTEXT, now=100,
                                     required_cohort=registry.cohort)
    assert ok, reason


def test_lower_tier_entity_cannot_enter_a_higher_cohort(group, venue):
    issuer, entities, _ = venue
    high = issuer.publish(cohort_id("JP", "bank", 4), registry_epoch=1, expires_at=9_999)
    with pytest.raises(ValueError):
        present(group, entities["GROUP-B"], high, scope=SCOPE, context=CONTEXT)


def test_presentation_does_not_transfer(group, venue):
    issuer, entities, registry = venue
    presentation = present(group, entities["GROUP-A"], registry, scope=SCOPE, context=CONTEXT)
    assert not verify_presentation(group, presentation, registry, issuer.public_key,
                                   scope="qomm:quote:epoch8", context=CONTEXT, now=100,
                                   required_cohort=registry.cohort)[0]
    assert not verify_presentation(group, presentation, registry, issuer.public_key,
                                   scope=SCOPE, context={"venue": "other"}, now=100,
                                   required_cohort=registry.cohort)[0]


def test_expired_or_foreign_registry_is_rejected(group, venue):
    issuer, entities, registry = venue
    presentation = present(group, entities["GROUP-A"], registry, scope=SCOPE, context=CONTEXT)
    assert not verify_presentation(group, presentation, registry, issuer.public_key,
                                   scope=SCOPE, context=CONTEXT, now=10_000,
                                   required_cohort=registry.cohort)[0]
    stranger = KybIssuer(group).public_key
    assert not verify_presentation(group, presentation, registry, stranger,
                                   scope=SCOPE, context=CONTEXT, now=100,
                                   required_cohort=registry.cohort)[0]


def test_registry_tampering_is_detected(group, venue):
    issuer, _, registry = venue
    intruder = group.base_pow(group.random_scalar())
    padded = SignedCohortRegistry(
        registry.cohort, registry.registry_epoch, registry.expires_at,
        registry.points + (intruder,), registry.issuer_public_key_hex,
        registry.registry_id, registry.signature_hex)
    ok, reason = verify_registry(group, padded, issuer.public_key, now=100)
    assert not ok and "contents" in reason


def test_cohort_gate_is_enforced_at_the_venue(group, venue):
    issuer, entities, registry = venue
    presentation = present(group, entities["GROUP-A"], registry, scope=SCOPE, context=CONTEXT)
    ok, reason = verify_presentation(group, presentation, registry, issuer.public_key,
                                     scope=SCOPE, context=CONTEXT, now=100,
                                     required_cohort=cohort_id("JP", "bank", 4))
    assert not ok and "cohort" in reason


def test_limits_bind_the_entity_not_the_wallet(group, venue):
    """Opening more wallets must not buy more allowance."""
    _, entities, _ = venue
    limiter = EntityRateLimiter(group, EntityLimits(max_requests=3, max_probe_lots=10_000,
                                                   max_epsilon=1.0))
    entity = entities["GROUP-A"]
    # every wallet of one entity derives the same scope nullifier
    nullifiers = [scope_nullifier(group, entity, SCOPE) for _ in range(4)]
    assert len({group.encode(n) for n in nullifiers}) == 1
    outcomes = [limiter.allow_request(nullifiers[i % 4], epoch=7)[0] for i in range(5)]
    assert outcomes == [True, True, True, False, False]
    # a different entity has its own allowance
    other = scope_nullifier(group, entities["GROUP-C"], SCOPE)
    assert limiter.allow_request(other, epoch=7)[0]


def test_probing_volume_and_privacy_budget_are_capped_per_entity(group, venue):
    _, entities, _ = venue
    limiter = EntityRateLimiter(group, EntityLimits(max_requests=100, max_probe_lots=100,
                                                   max_epsilon=0.5))
    nullifier = scope_nullifier(group, entities["GROUP-A"], SCOPE)
    assert limiter.allow_request(nullifier, 7, lots=60)[0]
    allowed, reason = limiter.allow_request(nullifier, 7, lots=60)
    assert not allowed and "volume" in reason
    assert limiter.spend_epsilon(nullifier, 7, 0.4)[0]
    assert not limiter.spend_epsilon(nullifier, 7, 0.4)[0]
    # a new epoch resets the allowance
    assert limiter.spend_epsilon(nullifier, 8, 0.4)[0]


def test_nullifier_separates_scopes_and_entities(group, venue):
    _, entities, _ = venue
    a7 = scope_nullifier(group, entities["GROUP-A"], SCOPE)
    a8 = scope_nullifier(group, entities["GROUP-A"], "qomm:quote:epoch8")
    c7 = scope_nullifier(group, entities["GROUP-C"], SCOPE)
    assert group.encode(a7) != group.encode(a8)
    assert group.encode(a7) != group.encode(c7)


def test_a_prover_that_skips_its_own_range_check_is_still_rejected(group):
    """The prover-side guard is convenience; the verifier is the security boundary."""
    from zk.commit import prove_range, verify_bounded, shift_commitment

    key = Pedersen(group, b"qomm:policy:v1")
    low, high = PolicyBounds().half          # (1, 200) -> 8 bits after shifting
    bits = max(1, (high - low).bit_length())
    out_of_band = 900
    blinding = key.random_blinding()
    commitment = key.commit(out_of_band, blinding)
    shifted = shift_commitment(key, commitment, low)
    # a malicious maker proves the truncated value, which is what fits the width
    truncated = (out_of_band - low) % (1 << bits)
    forged = prove_range(key, shifted, truncated, blinding, bits, b"ctx")
    assert not verify_bounded(key, commitment, forged, low, high, b"ctx")


def test_range_proof_rejects_a_relabelled_width(group):
    from zk.commit import prove_bounded, verify_bounded

    key = Pedersen(group, b"qomm:policy:v1")
    blinding = key.random_blinding()
    commitment, proof, bits = prove_bounded(key, 300, blinding, 0, 1023, b"ctx")
    assert verify_bounded(key, commitment, proof, 0, 1023, b"ctx")
    # the same proof must not pass as evidence for a tighter interval
    assert not verify_bounded(key, commitment, proof, 0, 255, b"ctx")
