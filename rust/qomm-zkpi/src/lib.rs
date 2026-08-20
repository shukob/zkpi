//! zkPI: a payment instruction that can be verified without being read.
//!
//! The point of the construction is that a settlement venue runs exactly two
//! operations --- verify the instruction, and check the nullifier has not been
//! seen --- and neither depends on how the price was reached. That makes it
//! pluggable: a venue with its own matching engine can accept instructions from
//! this issuer, or from any issuer speaking the same interface, without
//! adopting the rest of the design.
//!
//! The quorum signature is FROST rather than the threshold sigma protocol the
//! Python version carried. FROST is a real threshold Schnorr scheme with an
//! audit behind it; there was no reason to keep a hand-rolled one.

pub mod handles;

use curve25519_dalek::ristretto::RistrettoPoint;
use curve25519_dalek::scalar::Scalar;
use merlin::Transcript;
use qomm_zk::pedersen::Pedersen;
use qomm_zk::range::RangeCtx;
use rand_core::{CryptoRng, RngCore};
use sha2::{Digest, Sha512};
use std::collections::{BTreeMap, HashSet};

pub use frost_ristretto255 as frost;

/// The bounds a venue publishes in advance.
#[derive(Clone, Debug)]
pub struct Bounds {
    pub amount_bits: usize,
    pub price_bits: usize,
}

impl Default for Bounds {
    fn default() -> Self { Bounds { amount_bits: 32, price_bits: 32 } }
}

/// What the quorum issues. Nothing here reveals the trade.
#[derive(Clone)]
pub struct Instruction {
    pub amount_commitment: RistrettoPoint,
    pub price_commitment: RistrettoPoint,
    pub asset_commitment: RistrettoPoint,
    /// One aggregated range proof covering amount and price, rather than one
    /// each. Aggregation is the reason to be on Bulletproofs at all.
    pub ranges: bulletproofs::RangeProof,
    pub range_commitments: Vec<curve25519_dalek::ristretto::CompressedRistretto>,
    pub payer_handle: RistrettoPoint,
    pub payee_handle: RistrettoPoint,
    pub deadline: u64,
    pub nonce: [u8; 32],
    pub quote_key: u64,
    pub signature: frost::Signature,
}

/// What the issuer keeps: the openings, which the counterparties need.
pub struct Openings {
    pub amount: Scalar,
    pub price: Scalar,
    pub asset: Scalar,
}

impl Instruction {
    /// Binds the signature to the instruction; a signature cannot be replayed
    /// onto a different one.
    pub fn digest(&self) -> [u8; 64] {
        let mut hasher = Sha512::new();
        hasher.update(b"QOMM:ZKPI:v1");
        for point in [&self.amount_commitment, &self.price_commitment,
                      &self.asset_commitment, &self.payer_handle, &self.payee_handle] {
            hasher.update(point.compress().as_bytes());
        }
        hasher.update(self.deadline.to_be_bytes());
        hasher.update(self.nonce);
        hasher.update(self.quote_key.to_be_bytes());
        hasher.finalize().into()
    }

    /// Spent once. Derived from the nonce and the handles, so it reveals
    /// neither.
    pub fn nullifier(&self) -> [u8; 32] {
        let mut hasher = Sha512::new();
        hasher.update(b"QOMM:ZKPI:NUL:v1");
        hasher.update(self.nonce);
        hasher.update(self.payer_handle.compress().as_bytes());
        hasher.update(self.payee_handle.compress().as_bytes());
        let full: [u8; 64] = hasher.finalize().into();
        let mut out = [0u8; 32];
        out.copy_from_slice(&full[..32]);
        out
    }
}

pub struct Issuer {
    pub key: Pedersen,
    pub bounds: Bounds,
    pub ranges: RangeCtx,
}

impl Issuer {
    pub fn new(key: Pedersen, bounds: Bounds) -> Self {
        let bits = bounds.amount_bits.max(bounds.price_bits);
        let ranges = RangeCtx::new(bits, 2);
        Issuer { key, bounds, ranges }
    }

    /// Everything except the signature, which the quorum adds.
    #[allow(clippy::too_many_arguments)]
    pub fn build<R: RngCore + CryptoRng>(
        &self, amount: u64, price: u64, asset: u32,
        payer_handle: RistrettoPoint, payee_handle: RistrettoPoint,
        deadline: u64, nonce: [u8; 32], quote_key: u64, rng: &mut R,
    ) -> Result<(Vec<u8>, Openings, PartialInstruction), &'static str> {
        let openings = Openings {
            amount: Scalar::random(rng),
            price: Scalar::random(rng),
            asset: Scalar::random(rng),
        };
        let mut transcript = Transcript::new(b"qomm:zkpi:ranges");
        let (ranges, range_commitments) = self.ranges.prove(
            &mut transcript, &[amount, price], &[openings.amount, openings.price])?;
        let partial = PartialInstruction {
            amount_commitment: self.key.commit_u64(amount, &openings.amount),
            price_commitment: self.key.commit_u64(price, &openings.price),
            asset_commitment: self.key.commit_u64(asset as u64, &openings.asset),
            ranges, range_commitments,
            payer_handle, payee_handle, deadline, nonce, quote_key,
        };
        Ok((partial.digest().to_vec(), openings, partial))
    }
}

/// The instruction before the quorum has signed it.
pub struct PartialInstruction {
    pub amount_commitment: RistrettoPoint,
    pub price_commitment: RistrettoPoint,
    pub asset_commitment: RistrettoPoint,
    pub ranges: bulletproofs::RangeProof,
    pub range_commitments: Vec<curve25519_dalek::ristretto::CompressedRistretto>,
    pub payer_handle: RistrettoPoint,
    pub payee_handle: RistrettoPoint,
    pub deadline: u64,
    pub nonce: [u8; 32],
    pub quote_key: u64,
}

impl PartialInstruction {
    pub fn digest(&self) -> [u8; 64] {
        let mut hasher = Sha512::new();
        hasher.update(b"QOMM:ZKPI:v1");
        for point in [&self.amount_commitment, &self.price_commitment,
                      &self.asset_commitment, &self.payer_handle, &self.payee_handle] {
            hasher.update(point.compress().as_bytes());
        }
        hasher.update(self.deadline.to_be_bytes());
        hasher.update(self.nonce);
        hasher.update(self.quote_key.to_be_bytes());
        hasher.finalize().into()
    }

    pub fn sealed(self, signature: frost::Signature) -> Instruction {
        Instruction {
            amount_commitment: self.amount_commitment,
            price_commitment: self.price_commitment,
            asset_commitment: self.asset_commitment,
            ranges: self.ranges,
            range_commitments: self.range_commitments,
            payer_handle: self.payer_handle,
            payee_handle: self.payee_handle,
            deadline: self.deadline,
            nonce: self.nonce,
            quote_key: self.quote_key,
            signature,
        }
    }
}

/// The settlement side: verify, then spend once.
pub struct Venue {
    pub key: Pedersen,
    pub ranges: RangeCtx,
    pub group_public: frost::keys::PublicKeyPackage,
    spent: HashSet<[u8; 32]>,
}

impl Venue {
    pub fn new(key: Pedersen, bounds: &Bounds,
               group_public: frost::keys::PublicKeyPackage) -> Self {
        let bits = bounds.amount_bits.max(bounds.price_bits);
        Venue { key, ranges: RangeCtx::new(bits, 2), group_public, spent: HashSet::new() }
    }

    pub fn verify(&self, instruction: &Instruction, now: u64) -> Result<(), &'static str> {
        if now > instruction.deadline {
            return Err("past the deadline");
        }
        if self.spent.contains(&instruction.nullifier()) {
            return Err("already settled");
        }
        let mut transcript = Transcript::new(b"qomm:zkpi:ranges");
        if !self.ranges.verify(&mut transcript, &instruction.ranges,
                               &instruction.range_commitments) {
            return Err("amount or price outside the published bounds");
        }
        // the commitments the ranges cover must be the ones the instruction
        // carries, or the proof is about something else
        if instruction.range_commitments.first() != Some(&instruction.amount_commitment.compress())
            || instruction.range_commitments.get(1) != Some(&instruction.price_commitment.compress())
        {
            return Err("the range proof does not cover this instruction");
        }
        self.group_public
            .verifying_key()
            .verify(&instruction.digest(), &instruction.signature)
            .map_err(|_| "the quorum signature does not verify")?;
        Ok(())
    }

    pub fn settle(&mut self, instruction: &Instruction, now: u64) -> Result<(), &'static str> {
        self.verify(instruction, now)?;
        self.spent.insert(instruction.nullifier());
        Ok(())
    }

    pub fn spent(&self) -> usize { self.spent.len() }
}

/// A trusted-dealer quorum, which is what a computing node set is.
pub fn deal_quorum<R: RngCore + CryptoRng>(
    nodes: u16, threshold: u16, rng: &mut R,
) -> Result<(BTreeMap<frost::Identifier, frost::keys::SecretShare>,
             frost::keys::PublicKeyPackage), &'static str> {
    frost::keys::generate_with_dealer(
        nodes, threshold, frost::keys::IdentifierList::Default, rng)
        .map_err(|_| "dealing failed")
}
