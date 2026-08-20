//! One-out-of-many: the set says nothing about which member was used.

use curve25519_dalek::ristretto::RistrettoPoint;
use curve25519_dalek::scalar::Scalar;
use merlin::Transcript;
use qomm_zk::oneofmany::{prove, verify};
use qomm_zk::pedersen::Pedersen;
use rand::rngs::OsRng;

fn set(key: &Pedersen, size: usize, index: usize, rng: &mut OsRng)
    -> (Vec<RistrettoPoint>, Scalar) {
    let randomness = Scalar::random(rng);
    let commitments = (0..size)
        .map(|i| {
            if i == index { key.commit(&Scalar::ZERO, &randomness) }
            else { key.commit(&Scalar::random(rng), &Scalar::random(rng)) }
        })
        .collect();
    (commitments, randomness)
}

#[test]
fn a_member_that_opens_to_zero_can_be_proved_from_any_position() {
    let mut rng = OsRng;
    let key = Pedersen::new(b"qomm:gk:v1");
    for size in [2usize, 4, 8, 16] {
        for index in 0..size {
            let (commitments, randomness) = set(&key, size, index, &mut rng);
            let proof = prove(&key, &mut Transcript::new(b"t"), &commitments, index,
                              &randomness, &mut rng).expect("prove");
            assert!(verify(&key, &mut Transcript::new(b"t"), &commitments, &proof),
                    "size {size} index {index}");
        }
    }
}

#[test]
fn the_proof_does_not_vary_with_the_hidden_position() {
    let mut rng = OsRng;
    let key = Pedersen::new(b"qomm:gk:v1");
    let mut sizes = std::collections::HashSet::new();
    for index in 0..8 {
        let (commitments, randomness) = set(&key, 8, index, &mut rng);
        let proof = prove(&key, &mut Transcript::new(b"t"), &commitments, index,
                          &randomness, &mut rng).unwrap();
        assert!(verify(&key, &mut Transcript::new(b"t"), &commitments, &proof));
        sizes.insert(proof.size_bytes());
    }
    assert_eq!(sizes.len(), 1, "the hidden position leaks through the size");
}

#[test]
fn a_set_with_no_zero_member_cannot_be_proved() {
    let mut rng = OsRng;
    let key = Pedersen::new(b"qomm:gk:v1");
    let (mut commitments, randomness) = set(&key, 8, 3, &mut rng);
    let proof = prove(&key, &mut Transcript::new(b"t"), &commitments, 3,
                      &randomness, &mut rng).unwrap();
    commitments[3] = key.commit(&Scalar::from(1u64), &randomness);
    assert!(!verify(&key, &mut Transcript::new(b"t"), &commitments, &proof));
}

#[test]
fn a_proof_does_not_carry_over_to_another_set() {
    let mut rng = OsRng;
    let key = Pedersen::new(b"qomm:gk:v1");
    let (commitments, randomness) = set(&key, 8, 2, &mut rng);
    let proof = prove(&key, &mut Transcript::new(b"t"), &commitments, 2,
                      &randomness, &mut rng).unwrap();
    let (other, _) = set(&key, 8, 2, &mut rng);
    assert!(!verify(&key, &mut Transcript::new(b"t"), &other, &proof));
    assert!(!verify(&key, &mut Transcript::new(b"u"), &commitments, &proof));
}

#[test]
fn the_set_has_to_be_a_power_of_two() {
    let mut rng = OsRng;
    let key = Pedersen::new(b"qomm:gk:v1");
    let (commitments, randomness) = set(&key, 8, 0, &mut rng);
    assert!(prove(&key, &mut Transcript::new(b"t"), &commitments[..6], 0,
                  &randomness, &mut rng).is_err());
    assert!(prove(&key, &mut Transcript::new(b"t"), &commitments, 9,
                  &randomness, &mut rng).is_err());
}
