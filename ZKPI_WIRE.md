# zkPI on the wire, version 1

Big-endian throughout. Every field is fixed-width or length-prefixed, and the order below is the order on the wire. Nothing is optional: an instruction with a field missing is not a shorter instruction.

| field | bytes | what |
| --- | ---: | --- |
| magic | 8 | `QOMMZKPI` |
| version | 2 | currently 1. A verifier that meets one it does not know **stops** --- it does not guess at a layout, because a misparsed commitment is a valid point |
| amount commitment | 32 | compressed Ristretto |
| price commitment | 32 | compressed Ristretto |
| asset commitment | 32 | compressed Ristretto |
| payer handle | 32 | compressed Ristretto |
| payee handle | 32 | compressed Ristretto |
| deadline | 8 | seconds since the Unix epoch |
| nonce | 32 | what makes the nullifier unique |
| quote key | 8 | which quote the quorum priced |
| signature | 64 | FROST over Ristretto255 |
| range commitment count | 2 | how many commitments the range proofs are about |
| range commitments | 32 x count | compressed Ristretto each |
| amount proof length | 4 |  |
| amount range proof | that many | Bulletproofs |
| price proof length | 4 |  |
| price range proof | that many | Bulletproofs |

## What is deliberately absent

No compression, no self-describing container, no forward compatibility. Each is a way for two implementations to disagree about what they read, and the table above is a day's work to implement from.

## Checking an implementation against this

Vectors are in `artifacts/zkpi_vectors/`. One is a signed instruction and five
are it, damaged in the five ways a second implementation is most likely to get
wrong.

```
qomm-zkpi-verify --check-vectors artifacts/zkpi_vectors
```

The accepted vector must **decode and re-encode to the same bytes**. That is a
stronger statement than "it parsed": an implementation that dropped a field and
re-derived it would pass the second and fail the first, and it is the first that
decides whether two venues settle the same instruction.

Checking cannot compare against a freshly issued instruction --- a fresh one has
fresh randomness --- so the artefact on disk is the fixed point rather than the
generator. `--vectors <dir>` regenerates it, which is a thing to do when the
format changes and never because a check disagreed.

## Running the verifier

```
qomm-zkpi-verify --self-test                       # issue, encode, decode, verify
qomm-zkpi-verify --quorum <hex> --now <seconds> < instruction.bin
qomm-zkpi-verify < instruction.bin                 # layout only, and it says so
```

Exit 0 accepts, 1 rejects, 2 could not be asked. Without a quorum key it checks
the layout and says that is a parse and not a verification, because a tool that
printed "ok" for a well-formed unsigned instruction would be worse than none.

## Where this can run

A whole verification is about 1.6 ms of curve arithmetic, dominated by the two
Bulletproofs. Priced the way an EVM chain prices a precompile --- `ecrecover` is
3,000 gas for roughly 45 us of secp256k1, the closest reference point that
exists --- that is about **109,000 gas**.

Against it, `run_evm.py` measured **one ed25519 scalar multiplication written in
EVM bytecode at 302,401 gas**. So a whole verification as a precompile is
**cheaper than a single scalar multiplication simulated in 256-bit words**, and
one scalar multiplication is a small fraction of a verification, so the real
ratio is larger than the 2.8x that comparison shows.

**Which makes the deployment requirement specific rather than general: not
"which chain is cheap" but "which chain lets the curve be added".** Ethereum L1
does not --- its precompiles are fixed by consensus. An Avalanche L1 does,
because Subnet-EVM takes stateful precompiles as a build-time configuration, and
so does anything with a custom execution layer. Measured in
`artifacts/zkpi_wire.json`.
