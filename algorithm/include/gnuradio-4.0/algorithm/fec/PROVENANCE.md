# `algorithm/fec/` — provenance

This directory is forward error correction written for this project. **Nothing here is
carried from another codebase. Every file is a reimplementation**, and this document exists
to say so and to record why, per file, together with what each was checked against. The bias
that decided it is a standing one: reimplement where there is something to gain, carry where
there is not. The four kernels were authored in the gqrx4 tree's P25 strand and moved here
by their author's direction; the checks below were made there and travel with the code as
its qa suites.

## What each file is, and why it was written rather than lifted

| File              | Code                                                                              | Candidate source considered                                                                    | Decision      |
| ----------------- | --------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | ------------- |
| `Golay.hpp`       | Golay(24,12,8), (23,12,7) and the shortened (18,6,8)                              | liquid-dsp `src/fec/src/fec_golay2412.c` (MIT, 403 lines)                                      | reimplemented |
| `Hamming.hpp`     | Hamming(15,11,3) and the shortened (10,6,3)                                       | liquid-dsp `src/fec/src/fec_hamming1511.c` (MIT, 136 lines)                                    | reimplemented |
| `ReedSolomon.hpp` | RS over GF(2^M); the driving GF(64) codes RS(24,12,13), RS(24,16,9), RS(36,20,17) | GNU Radio `gr-fec/lib/reed-solomon/` (Karn)                                                    | reimplemented |
| `Bch.hpp`         | binary BCH(63,16,23)                                                              | none carried a fit; written from the standard's stated field, polynomial and designed distance | reimplemented |

**Golay.** The lift was expected to be the cheap answer and stopped being one on contact with
what P25 actually needs. The part inventory names Golay(23,12,7), the form the vocoder frame
uses; the payload layer needs the shortened **(18,6,8)**, which liquid does not provide at
all, and a lift would therefore have been a lift plus a shim plus a second decoder. Deriving
the one code all three forms are shortenings and extensions of costs about 60 lines of
substance, and it comes with a property no carried table has: the coset table is built by
enumerating the error patterns it must cover, so building it either fills all 2048 slots
exactly once or the generator is wrong. That check is asserted in the qa.

**Hamming.** Two codes are needed, (15,11,3) for the vocoder frame and (10,6,3) for the
payload hexbits, and liquid supplies one of them. Both reduce to a list of parity columns
and a sixteen-entry syndrome lookup, so the file states the two column lists and shares one
decoder between them. The carried alternative was 136 lines for half the requirement.

**Reed-Solomon.** This is the one where reimplementation had something to optimize, and it
is also the one where the risk of getting it wrong was highest — so it was cross-checked
hardest. Karn's decoder is generic over symbol sizes 1 through 8 and reaches its field
through a heap-allocated control block built at run time; the driving consumer needs exactly
one field, GF(64) on x^6 + x + 1, at three parity lengths. A header-only template over the
field and the parity count with compile-time tables removes the allocation, the indirection
and the run-time parameterization together. Two behaviors were added that neither Karn's
decoder nor the behavioral oracle has, both of which turn a silent wrong answer into a
reported failure: a correction landing inside the shortening's padding is refused rather
than skipped, and the syndromes are recomputed after correction so that a locator polynomial
which does not describe the received word cannot return symbols. The generator's first root
is alpha^1 throughout — the convention every validated instantiation shares; a code family
rooted elsewhere brings its own oracle when it brings its first consumer.

**BCH.** The generator polynomial is **derived, not tabulated**: everything it follows from
— the field, the primitive polynomial, the designed distance — is stated by TIA-102.BAAA,
while the degree-47 polynomial itself is a consequence nobody can check by eye, so the
derivation through the cyclotomic cosets is the artifact and a copied constant would have
been the risk. Decoding is exhaustive nearest-codeword search over the 65536 codewords
rather than an algebraic decoder, because at network-identifier cadence an algebraic decoder
buys nothing and an exhaustive search cannot be wrong about which codeword is nearest; the
minimum distance of 23 makes the first codeword found within 11 bits provably the unique
nearest. The code was verified in its authoring tree against captured air traffic — real
network identifiers decoded off a 483.125 MHz P25 repeater — in addition to the property
sweeps the qa carries.

## What each was checked against

Correctness here rests on independent implementations, not on the tests agreeing with the
code that wrote them.

- **Reed-Solomon** was run against Karn's own decoder at all three GF(64) configurations,
  20,000 randomized codewords each. At and under every code's correction limit: **identical
  encoder parity, identical corrected words, identical accept/reject verdicts, zero
  disagreements in 60,000 trials.** Past the limit the two diverge in one direction only —
  Karn accepted a miscorrected payload where this decoder reported a failure, never the
  reverse. One parity vector per code is pinned as a literal in the qa, so the field, the
  generator, the first root and the shortening are all held against a value this tree did
  not compute.
- **Golay** was run against a behavioral oracle's decoder over all 4096 codewords: encoding
  identical. Decoding was checked exhaustively rather than sampled — all 4096 codewords
  against all 11,980,800 error patterns of weight three or less, zero failures — and all
  680,064 weight-four patterns are reported rather than mis-decoded.
- **Hamming** encoding is identical to the oracle's over all 2048 and all 64 words
  respectively, and single-error correction is exhaustive for both.
- **BCH** is held by the qa's exhaustive weight sweep over all 65536 codewords (minimum
  distance exactly 23), linearity and systematic-form properties, per-weight correction
  sweeps through the full radius of 11, a measured false-accept rate on random words, and
  the air-traffic verification recorded above.

The exhaustive sweeps live in the qa tests, so they are re-run by the suite rather than
being a one-time claim.
