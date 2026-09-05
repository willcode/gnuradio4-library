# `algorithm/vocoder/` — provenance

This directory is a vocoder written for this project from a specification, not carried from
another codebase. `Imbe.hpp` is a **translation of the owner's own reference implementation**
of a corpus note that states the IMBE 7200x4400 decoder in real arithmetic; it is not a port
of any existing decoder, and this document exists to say precisely what was used, what was
deliberately not used, and what the result was checked against. The distinction matters more
here than for a FEC kernel, because a working decoder for this codec does exist under the GPL
and is carried elsewhere in this project's world — so the line between the standard's facts
and one author's expression of them is the whole subject of this file.

## What the file is, and where each part comes from

| Layer                                                                               | Source                                                               | Status               |
| ----------------------------------------------------------------------------------- | -------------------------------------------------------------------- | -------------------- |
| The algorithm, its constants, tables, bit layouts and equations                     | TIA-102.BABA, _Project 25 vocoder description_                       | **normative source** |
| The real-arithmetic specification of the decoder                                    | the v5 corpus note `imbe-vocoder` (`corpus/notes/imbe-vocoder.yaml`) | **specification**    |
| The code this header was translated from                                            | the owner's reference implementation `v5/gr5/imbe.py`                | **translated**       |
| A fixed-point realization (Pavel Yazev's, via op25, GPL, carried in the gqrx4 tree) | —                                                                    | **not carried**      |

**TIA-102.BABA is the normative source.** Every constant and formula in the header is the
standard's: the harmonic-count rule, the bit allocation table, the gain quantizer and its step
maps, the residual step sizes, the inverse DCT, the prediction blend, the enhancement formula,
the Park-Miller noise generator, the synthesis window, the band-bin edges.

**The v5 note is the specification.** It states the decoder in real arithmetic, pins everything
exactly pinnable — bit fields, integer rules, the noise generator and its consumption order,
window values, band regions — and grants numeric freedom elsewhere. Its rule names are the
vocabulary of this header's doc comments (`b0_rule`, `rescanning_rule`,
`prng_consumption_order`, `voiced_case_rule`, `overlap_add_rule`, ...), so any statement in the
code can be traced back to the sentence that authorizes it. The note was verified before it was
used: its thirteen pinned vectors — the PRNG sequence and its two consumption anchors, the L
staircase, the unpack probe and its one-bit-move property, the three on-air parameter sets, the
amplitude chain through the prediction re-blend, determinism, and the concealment advance — all
reproduce through the reference implementation.

**`v5/gr5/imbe.py` is what was translated.** The header follows it statement for statement,
including the arithmetic order the note pins as codec state: which harmonic count feeds `Lmax`
in the phase stage (the synthesizer's previous count, never the predictor's), which feeds the
prediction interpolation (the predictor's, initially 30), and the exact order of the three draw
sites. Those are not stylistic details — two implementations that draw in different orders
produce different speech from identical input.

## The licensing line

Stated once, because it decided the shape of the file.

The constants, tables, layouts and equations are **the standard's facts**, and carrying them is
what this project does. A fixed-point **realization** of them is something else: its saturating
arithmetic, its table-interpolated `cos`/`log2`/`pow2`/`sqrt`, and its FFT's per-stage scaling
are an author's expression, and a sample-exact clone of them would be that author's program
wearing new syntax, with that license's obligations.

None of that realization is carried here. Specifically:

- **no saturating arithmetic** — the internals are `double`, and the only saturation is the one
  the codec's output contract itself requires, at the int16 rails on the way out;
- **no table-interpolated transcendentals** — `std::cos`, `std::exp2`, `std::pow` and
  `std::sqrt` are called directly;
- **no per-stage-scaled FFT** — the inverse DCT is the direct O(M²) sum (M is at most ten) and
  the unvoiced realization is the direct 256-point sum over the at-most-128 active bins.

The consequence is stated as plainly as the rule: **sample-exact equality with the fixed-point
decoder is not a conformance criterion and is not claimed.** Conformance is stated structurally
and functionally — packing, determinism, state advance, boundary and transition behavior — and
those are what `qa_Imbe` asserts.

## What it was checked against

Parity with the fixed-point decoder is **measured, not asserted**. The note's recorded
measurement, taken over the committed off-air fixture's **234 vocoded codewords** with both
decoders fresh and fed the same parameter words in transmitted order:

- zero-lag correlation **0.998** over the full 37,440-sample streams, whole-stream levels
  **0.02 dB** apart;
- over the 49 codewords louder than -40 dBFS: per-codeword levels within **0.31 dB** worst and
  **0.04 dB** mean, per-codeword correlation median **0.998**, minimum **0.997**;
- no codeword loud in one decoder is quiet in the other, and the exactly-silent codewords agree
  to within a couple of least-significant samples.

The residual is float against table-driven fixed point. The measurement is the fixture plus the
two decoders, so it reproduces anywhere — and it earns its keep: its first run is what caught a
unit convention in the enhancement denominator (the fundamental there is cycles per sample, not
radians), which had been costing about 2 dB of formant level on every loud frame. That is an
A/B doing exactly the job the licensing line assigns it.

Against the reference implementation the header was translated from, the check is direct
equality of decoded PCM on the fixtures the qa carries — the same equations in both languages,
both in float64 — with the qa asserting the note's pinned vectors rather than a golden stream,
because last-ulp arithmetic is what floats across platforms while structure and determinism do
not.
