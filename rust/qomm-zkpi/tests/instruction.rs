//! An instruction a venue can check without reading, signed by a quorum.

use curve25519_dalek::ristretto::RistrettoPoint;
use curve25519_dalek::scalar::Scalar;
use qomm_zk::pedersen::Pedersen;
use qomm_zkpi::{deal_quorum, frost, Bounds, Issuer, Venue};
use rand::rngs::OsRng;
use std::collections::BTreeMap;

const NODES: u16 = 7;
const THRESHOLD: u16 = 3;

struct Quorum {
    shares: BTreeMap<frost::Identifier, frost::keys::KeyPackage>,
    public: frost::keys::PublicKeyPackage,
}

fn quorum(rng: &mut OsRng) -> Quorum {
    let (secret, public) = deal_quorum(NODES, THRESHOLD, rng).expect("deal");
    let shares = secret
        .into_iter()
        .map(|(id, share)| (id, frost::keys::KeyPackage::try_from(share).unwrap()))
        .collect();
    Quorum { shares, public }
}

/// Two rounds of FROST, as the nodes would run them.
fn sign(q: &Quorum, message: &[u8], signers: usize, rng: &mut OsRng) -> frost::Signature {
    try_sign(q, message, signers, rng).expect("aggregate")
}

fn try_sign(q: &Quorum, message: &[u8], signers: usize, rng: &mut OsRng)
    -> Result<frost::Signature, ()> {
    let chosen: Vec<_> = q.shares.keys().take(signers).cloned().collect();
    let mut nonces = BTreeMap::new();
    let mut commitments = BTreeMap::new();
    for id in &chosen {
        let (nonce, commitment) =
            frost::round1::commit(q.shares[id].signing_share(), rng);
        nonces.insert(*id, nonce);
        commitments.insert(*id, commitment);
    }
    let package = frost::SigningPackage::new(commitments, message);
    let mut shares = BTreeMap::new();
    for id in &chosen {
        // round 2 already refuses a package that names too few signers
        let share = frost::round2::sign(&package, &nonces[id], &q.shares[id])
            .map_err(|_| ())?;
        shares.insert(*id, share);
    }
    frost::aggregate(&package, &shares, &q.public).map_err(|_| ())
}

fn issue(rng: &mut OsRng, q: &Quorum, amount: u64, price: u64, deadline: u64)
    -> (qomm_zkpi::Instruction, qomm_zkpi::Openings) {
    let key = Pedersen::new(b"qomm:defmi:v1");
    let issuer = Issuer::new(key, Bounds::default());
    let (digest, openings, partial) = issuer
        .build(amount, price, 3,
               RistrettoPoint::mul_base(&Scalar::from(11u64)),
               RistrettoPoint::mul_base(&Scalar::from(22u64)),
               deadline, [7u8; 32], 1_599_845, rng)
        .expect("build");
    let signature = sign(q, &digest, THRESHOLD as usize, rng);
    (partial.sealed(signature), openings)
}

fn venue(q: &Quorum) -> Venue {
    Venue::new(Pedersen::new(b"qomm:defmi:v1"), &Bounds::default(), q.public.clone())
}

#[test]
fn a_signed_instruction_verifies_and_settles_once() {
    let mut rng = OsRng;
    let q = quorum(&mut rng);
    let (instruction, _) = issue(&mut rng, &q, 100, 99_990, 1_500);
    let mut v = venue(&q);
    assert!(v.verify(&instruction, 1_000).is_ok());
    assert!(v.settle(&instruction, 1_000).is_ok());
    assert_eq!(v.settle(&instruction, 1_000), Err("already settled"));
    assert_eq!(v.spent(), 1);
}

#[test]
fn a_second_venue_accepts_the_same_instruction() {
    // pluggability is the property, so it gets a test
    let mut rng = OsRng;
    let q = quorum(&mut rng);
    let (instruction, _) = issue(&mut rng, &q, 100, 99_990, 1_500);
    assert!(venue(&q).verify(&instruction, 1_000).is_ok());
    assert!(venue(&q).verify(&instruction, 1_000).is_ok());
}

#[test]
fn an_expired_instruction_is_refused() {
    let mut rng = OsRng;
    let q = quorum(&mut rng);
    let (instruction, _) = issue(&mut rng, &q, 100, 99_990, 1_500);
    assert_eq!(venue(&q).verify(&instruction, 2_000), Err("past the deadline"));
}

#[test]
fn a_tampered_field_breaks_the_signature() {
    let mut rng = OsRng;
    let q = quorum(&mut rng);
    let (mut instruction, _) = issue(&mut rng, &q, 100, 99_990, 1_500);
    instruction.quote_key += 1;
    assert_eq!(venue(&q).verify(&instruction, 1_000),
               Err("the quorum signature does not verify"));
}

#[test]
fn a_signature_does_not_transfer_to_another_instruction() {
    let mut rng = OsRng;
    let q = quorum(&mut rng);
    let (first, _) = issue(&mut rng, &q, 100, 99_990, 1_500);
    let (mut second, _) = issue(&mut rng, &q, 200, 99_990, 1_500);
    second.signature = first.signature;
    assert!(venue(&q).verify(&second, 1_000).is_err());
}

#[test]
fn a_range_proof_from_elsewhere_does_not_cover_this_instruction() {
    let mut rng = OsRng;
    let q = quorum(&mut rng);
    let (mut instruction, _) = issue(&mut rng, &q, 100, 99_990, 1_500);
    let (other, _) = issue(&mut rng, &q, 5, 7, 1_500);
    instruction.ranges = other.ranges;
    instruction.range_commitments = other.range_commitments;
    assert!(venue(&q).verify(&instruction, 1_000).is_err());
}

#[test]
fn fewer_signers_than_the_threshold_cannot_sign() {
    // FROST refuses to aggregate rather than producing a signature that fails
    // later, which is the stronger of the two behaviours.
    let mut rng = OsRng;
    let q = quorum(&mut rng);
    assert!(try_sign(&q, b"anything", (THRESHOLD - 1) as usize, &mut rng).is_err());
    assert!(try_sign(&q, b"anything", THRESHOLD as usize, &mut rng).is_ok());
}

#[test]
fn an_amount_outside_the_published_bounds_cannot_be_issued() {
    let mut rng = OsRng;
    let issuer = Issuer::new(Pedersen::new(b"qomm:defmi:v1"), Bounds::default());
    assert!(issuer
        .build(1u64 << 40, 99_990, 3,
               RistrettoPoint::mul_base(&Scalar::from(11u64)),
               RistrettoPoint::mul_base(&Scalar::from(22u64)),
               1_500, [7u8; 32], 1, &mut rng)
        .is_err());
}

/// A deadline the venue will not hold a nullifier for is refused.
///
/// The bounded-state argument is that a nullifier only has to outlive its
/// deadline, so state grows with instructions in flight rather than with
/// history. That needs an upper bound on how far out a deadline may be, and
/// only "not expired yet" was checked: an instruction dated a century out kept
/// its nullifier for a century, and the argument did not hold.
#[test]
fn a_deadline_beyond_the_venues_horizon_is_refused() {
    let mut rng = OsRng;
    let q = quorum(&mut rng);
    let bounds = Bounds::default();
    let far = 1_000 + bounds.max_horizon + 1;
    let (instruction, _) = issue(&mut rng, &q, 100, 99_990, far);
    assert_eq!(venue(&q).verify(&instruction, 1_000),
               Err("the deadline is further out than this venue will hold a \
nullifier for"));
}

#[test]
fn a_deadline_inside_the_horizon_is_accepted() {
    let mut rng = OsRng;
    let q = quorum(&mut rng);
    let near = 1_000 + Bounds::default().max_horizon;
    let (instruction, _) = issue(&mut rng, &q, 100, 99_990, near);
    assert!(venue(&q).verify(&instruction, 1_000).is_ok());
}

/// Declaring a narrow amount and a wide price bought nothing: one aggregated
/// range proof covers both fields at one width, so the amount was proved at
/// the price's width. The venue refuses the pair rather than accepting the
/// wider and calling it the narrower.
#[test]
fn a_venue_will_not_pretend_one_range_proof_shows_two_widths() {
    let mut rng = OsRng;
    let q = quorum(&mut rng);
    let bounds = Bounds { amount_bits: 24, price_bits: 64, ..Bounds::default() };
    let (instruction, _) = issue(&mut rng, &q, 100, 99_990, 1_500);
    let mut mixed = Venue::new(Pedersen::new(b"qomm:zkpi:v1"), &bounds,
                               q.public.clone());
    assert_eq!(mixed.verify(&instruction, 1_000),
               Err("this venue declares different widths for amount and price, \
and one aggregated range proof cannot show two widths"));
}
