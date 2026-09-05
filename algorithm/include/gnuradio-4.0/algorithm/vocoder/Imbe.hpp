#ifndef GNURADIO_ALGORITHM_VOCODER_IMBE_HPP
#define GNURADIO_ALGORITHM_VOCODER_IMBE_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <optional>
#include <span>

/**
 * IMBE 7200x4400, the multi-band excitation vocoder P25 Phase 1 carries (TIA-102.BABA). One
 * codeword in, one frame out: the eight parameter words u0..u7 of a 20 ms voice codeword
 * become 160 samples of 8 kHz speech. 144 transmitted bits carry 88 information bits, which
 * is where the two rates in the name come from.
 *
 * The decoder is stated in real arithmetic. What the standard pins exactly is exact here —
 * the bit fields, the allocation and quantizer tables, the Park-Miller noise generator and
 * its consumption order, the window values, the band-bin edges — and the synthesis math is
 * `double` throughout. That is a deliberate boundary: the constants, tables, layouts and
 * equations are the standard's facts, while a fixed-point realization of them (saturating
 * arithmetic, table-interpolated transcendentals, a per-stage-scaled FFT) is an author's
 * expression and is not carried. `PROVENANCE.md` beside this header documents that boundary.
 *
 * Speech state carries between codewords and belongs to the decoder, so codewords are decoded
 * in transmitted order, one `ImbeDecoder` instance serves one voice stream, and `reset()` is
 * exactly construction. Decoding is deterministic: the noise generator is integer and its
 * consumption order is part of the codec state, so the same codewords produce the same samples
 * on every run and every schedule.
 *
 * Everything outside the codec — the codeword's FEC layers, the parameter-word widths, the
 * audio rate, any makeup gain — belongs to the protocol layer. This header spans exactly the
 * eight words going in and the 160 samples coming out.
 */
namespace gr::vocoder {

//! Parameter words per codeword, widths 12, 12, 12, 12, 11, 11, 11, 7 (the protocol layer's
//! word boundaries).
inline constexpr std::size_t kImbeParameterWords = 8UZ;

//! Samples one codeword decodes to: 20 ms at 8000 samples a second.
inline constexpr std::size_t kImbeSamplesPerFrame = 160UZ;

//! Harmonic count bounds. L is 9 at b0 = 0 and 56 at b0 = 204..207, never outside that range.
inline constexpr unsigned kImbeMinHarmonics = 9U;
inline constexpr unsigned kImbeMaxHarmonics = 56U;

//! The voicing-band count saturates at twelve, which binds exactly when L > 36.
inline constexpr unsigned kImbeMaxBands = 12U;

//! Spectral bins of the unvoiced synthesizer's transform.
inline constexpr std::size_t kImbeNoiseBins = 256UZ;

//! Samples of the unvoiced realization held over to the next frame.
inline constexpr std::size_t kImbeUnvoicedTail = 105UZ;

//! Log-amplitude vector, harmonic-indexed: entry `l` is harmonic `l`, entry 0 unused. Sized
//! for the largest L plus the one entry past it the prediction interpolation reads.
using ImbeAmplitudes = std::array<double, kImbeMaxHarmonics + 1UZ>;

// ----------------------------------------------------------------------- the noise generator

/**
 * The one Park-Miller generator that drives every random draw in the codec.
 *
 * Its state and its consumption order are codec state: two decoders that draw in different
 * orders produce different speech from identical input, so every draw site below states where
 * it falls in that order directly.
 *
 * From seed 1 the first twelve outputs, times 32768, are 16807, 15089, -21287, 3114, -18558,
 * -9528, -28968, 2558, 12099, 1101, -26472, 15445.
 */
struct ImbeRng {
    std::uint32_t seed{1U};

    //! Advances the state and returns the new seed's low sixteen bits, read as a signed
    //! integer and divided by 32768 — a real in (-1, 1).
    [[nodiscard]] double draw() noexcept {
        seed                    = static_cast<std::uint32_t>((16807ULL * static_cast<std::uint64_t>(seed)) % 2147483647ULL);
        const std::uint16_t low = static_cast<std::uint16_t>(seed & 0xFFFFU);
        return static_cast<double>(static_cast<std::int16_t>(low)) / 32768.0;
    }
};

// ------------------------------------------------------------------- integer parameter rules

//! L, the number of harmonics, from the pitch index alone: an exact integer rule in which
//! 60647/65536 is the standard's 0.9254.
[[nodiscard]] inline constexpr unsigned imbeHarmonicCount(unsigned b0) noexcept { return (60647U * ((2U * b0 + 81U) / 8U)) >> 16U; }

//! B, the number of voicing bands: floor((L + 2)/3), capped at twelve.
[[nodiscard]] inline constexpr unsigned imbeBandCount(unsigned L) noexcept { return std::min((L + 2U) / 3U, kImbeMaxBands); }

//! The fundamental in radians per sample: a pitch period of (b0 + 39.5)/2 samples, 19.75 to
//! 123.25, about 405 Hz down to 65 Hz.
[[nodiscard]] inline constexpr double imbeFundamental(unsigned b0) noexcept { return 4.0 * std::numbers::pi / (static_cast<double>(b0) + 39.5); }

// -------------------------------------------------------------------------------- the tables

namespace detail {

//! Longest allocation row, L - 1 entries at L = 56.
inline constexpr std::size_t kAllocationStride = 55UZ;

/**
 * For each L, how many payload bits each of b3..b(L+1) receives. No generating rule is known,
 * so the 48 rows are carried verbatim, L - 1 entries per row, zero-padded to the stride. Row L
 * lives at index L - 9.
 */
inline constexpr std::array<std::array<std::uint8_t, kAllocationStride>, 48UZ> kAllocationRows{{
    {10, 9, 9, 9, 9, 9, 8, 7},                                                                                                                                             // L = 9
    {9, 9, 8, 8, 8, 9, 7, 6, 5},                                                                                                                                           // L = 10
    {8, 8, 8, 7, 7, 9, 7, 6, 5, 4},                                                                                                                                        // L = 11
    {8, 7, 7, 7, 7, 8, 7, 6, 5, 4, 3},                                                                                                                                     // L = 12
    {7, 7, 7, 6, 6, 7, 7, 6, 5, 4, 3, 3},                                                                                                                                  // L = 13
    {7, 6, 6, 6, 6, 7, 7, 5, 4, 4, 3, 4, 3},                                                                                                                               // L = 14
    {7, 6, 6, 6, 5, 6, 7, 5, 4, 4, 3, 3, 3, 3},                                                                                                                            // L = 15
    {6, 6, 6, 5, 5, 6, 6, 5, 4, 4, 3, 3, 3, 3, 2},                                                                                                                         // L = 16
    {6, 6, 5, 5, 5, 5, 5, 5, 4, 4, 4, 3, 3, 2, 3, 2},                                                                                                                      // L = 17
    {6, 5, 5, 5, 5, 5, 4, 5, 5, 4, 3, 3, 3, 3, 2, 2, 2},                                                                                                                   // L = 18
    {6, 5, 5, 4, 4, 5, 4, 5, 4, 4, 3, 3, 3, 3, 2, 3, 2, 1},                                                                                                                // L = 19
    {6, 5, 5, 4, 4, 5, 4, 5, 4, 4, 3, 3, 2, 3, 2, 1, 3, 2, 1},                                                                                                             // L = 20
    {5, 5, 5, 4, 4, 4, 4, 5, 4, 4, 3, 3, 2, 2, 3, 2, 1, 3, 2, 1},                                                                                                          // L = 21
    {5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 3, 2, 3, 2, 2, 3, 2, 1, 2, 2, 1},                                                                                                       // L = 22
    {5, 4, 4, 4, 4, 4, 3, 4, 4, 3, 4, 3, 2, 3, 2, 2, 2, 2, 1, 2, 2, 1},                                                                                                    // L = 23
    {5, 4, 4, 4, 4, 4, 3, 3, 4, 3, 3, 3, 3, 2, 3, 2, 1, 2, 2, 1, 2, 2, 1},                                                                                                 // L = 24
    {5, 4, 4, 4, 3, 4, 3, 3, 4, 3, 3, 3, 3, 2, 3, 2, 1, 2, 2, 1, 2, 1, 1, 1},                                                                                              // L = 25
    {5, 4, 4, 3, 3, 4, 3, 3, 4, 3, 3, 3, 2, 2, 3, 2, 1, 2, 2, 1, 1, 2, 2, 1, 1},                                                                                           // L = 26
    {5, 4, 4, 3, 3, 4, 3, 2, 4, 3, 2, 3, 2, 2, 3, 2, 2, 1, 2, 2, 1, 1, 2, 2, 1, 1},                                                                                        // L = 27
    {4, 4, 4, 3, 3, 4, 3, 2, 4, 3, 2, 3, 2, 2, 2, 3, 2, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1},                                                                                     // L = 28
    {4, 4, 4, 3, 3, 3, 3, 2, 4, 3, 2, 2, 3, 2, 2, 2, 3, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1},                                                                                  // L = 29
    {4, 4, 4, 3, 3, 3, 3, 2, 2, 3, 3, 2, 2, 3, 2, 2, 1, 3, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1},                                                                               // L = 30
    {4, 4, 3, 3, 3, 3, 3, 2, 2, 3, 3, 2, 2, 3, 2, 2, 1, 2, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1},                                                                            // L = 31
    {4, 4, 3, 3, 3, 3, 3, 2, 2, 3, 3, 2, 2, 3, 2, 2, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 0},                                                                         // L = 32
    {4, 3, 3, 3, 3, 3, 3, 2, 2, 3, 3, 2, 2, 3, 2, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1},                                                                      // L = 33
    {4, 3, 3, 3, 3, 3, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 0},                                                                   // L = 34
    {4, 3, 3, 3, 3, 3, 2, 2, 2, 3, 2, 2, 2, 2, 3, 2, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 0, 2, 1, 1, 1, 0},                                                                // L = 35
    {4, 3, 3, 3, 3, 3, 2, 2, 2, 1, 3, 2, 2, 2, 1, 3, 2, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 0, 2, 1, 1, 1, 0},                                                             // L = 36
    {4, 3, 3, 3, 2, 3, 2, 2, 2, 1, 3, 2, 2, 2, 2, 3, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 0, 2, 1, 1, 1, 1, 0},                                                          // L = 37
    {4, 3, 3, 3, 2, 3, 2, 2, 2, 1, 3, 2, 2, 2, 1, 3, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 0, 2, 1, 1, 1, 1, 0},                                                       // L = 38
    {4, 3, 3, 3, 2, 3, 2, 2, 2, 1, 3, 2, 2, 2, 1, 3, 2, 1, 1, 1, 2, 2, 1, 1, 1, 0, 2, 1, 1, 1, 1, 0, 2, 1, 1, 1, 0, 0},                                                    // L = 39
    {4, 3, 3, 3, 2, 3, 2, 2, 2, 1, 3, 2, 2, 1, 1, 3, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 0, 2, 1, 1, 1, 1, 0, 2, 1, 1, 1, 0, 0},                                                 // L = 40
    {4, 3, 3, 2, 2, 3, 2, 2, 1, 1, 3, 2, 2, 2, 1, 1, 3, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 0, 2, 1, 1, 1, 1, 0, 2, 1, 1, 1, 0, 0},                                              // L = 41
    {4, 3, 3, 2, 2, 3, 2, 2, 2, 1, 1, 3, 2, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 0, 2, 1, 1, 1, 0, 0, 2, 1, 1, 1, 0, 0},                                           // L = 42
    {4, 3, 3, 2, 2, 3, 2, 2, 2, 1, 1, 3, 2, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 0, 2, 1, 1, 1, 0, 0, 2, 1, 1, 1, 1, 0, 0},                                        // L = 43
    {4, 3, 3, 2, 2, 3, 2, 2, 1, 1, 1, 3, 2, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 0, 2, 1, 1, 1, 1, 0, 0, 2, 1, 1, 1, 1, 0, 0},                                     // L = 44
    {4, 3, 3, 2, 2, 3, 2, 2, 1, 1, 1, 3, 2, 2, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 0, 0, 2, 1, 1, 1, 1, 0, 0, 2, 1, 1, 1, 1, 0, 0},                                  // L = 45
    {3, 3, 3, 2, 2, 3, 2, 2, 1, 1, 1, 3, 2, 2, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 0, 0, 2, 1, 1, 1, 1, 0, 0, 2, 1, 1, 1, 1, 0, 0},                               // L = 46
    {3, 3, 3, 2, 2, 3, 2, 2, 1, 1, 1, 3, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 0, 0, 2, 1, 1, 1, 1, 0, 0, 2, 1, 1, 1, 0, 0, 0},                            // L = 47
    {3, 3, 3, 2, 2, 3, 2, 2, 1, 1, 1, 1, 3, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 0, 0, 2, 1, 1, 1, 0, 0, 0, 2, 1, 1, 1, 0, 0, 0},                         // L = 48
    {3, 3, 3, 2, 2, 3, 2, 2, 1, 1, 1, 1, 3, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 0, 2, 2, 1, 1, 1, 0, 0, 2, 1, 1, 1, 0, 0, 0, 2, 1, 1, 1, 1, 0, 0, 0},                      // L = 49
    {3, 3, 3, 2, 2, 3, 2, 2, 1, 1, 1, 1, 3, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 0, 2, 2, 1, 1, 1, 0, 0, 2, 1, 1, 1, 1, 0, 0, 0, 2, 1, 1, 1, 0, 0, 0, 0},                   // L = 50
    {3, 3, 3, 2, 2, 3, 2, 2, 1, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 0, 2, 2, 1, 1, 1, 0, 0, 0, 2, 1, 1, 1, 1, 0, 0, 0, 2, 1, 1, 1, 1, 0, 0, 0},                // L = 51
    {3, 3, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1, 3, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 0, 2, 2, 1, 1, 1, 0, 0, 0, 2, 1, 1, 1, 1, 0, 0, 0, 2, 1, 1, 1, 1, 0, 0, 0},             // L = 52
    {3, 3, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1, 3, 2, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 0, 2, 2, 1, 1, 1, 0, 0, 0, 2, 1, 1, 1, 1, 0, 0, 0, 2, 1, 1, 1, 0, 0, 0, 0},          // L = 53
    {3, 3, 2, 2, 2, 3, 2, 2, 1, 1, 1, 1, 0, 3, 2, 2, 1, 1, 1, 1, 0, 2, 2, 1, 1, 1, 1, 1, 0, 2, 2, 1, 1, 1, 0, 0, 0, 2, 1, 1, 1, 1, 0, 0, 0, 2, 1, 1, 1, 0, 0, 0, 0},       // L = 54
    {3, 3, 2, 2, 2, 3, 2, 2, 1, 1, 1, 1, 0, 3, 2, 2, 1, 1, 1, 1, 0, 2, 2, 1, 1, 1, 1, 1, 0, 2, 2, 1, 1, 1, 0, 0, 0, 2, 1, 1, 1, 0, 0, 0, 0, 2, 1, 1, 1, 1, 0, 0, 0, 0},    // L = 55
    {3, 3, 2, 2, 2, 3, 2, 2, 1, 1, 1, 1, 0, 3, 2, 2, 1, 1, 1, 1, 0, 2, 2, 1, 1, 1, 1, 1, 0, 2, 2, 1, 1, 1, 0, 0, 0, 2, 1, 1, 1, 1, 0, 0, 0, 0, 2, 1, 1, 1, 0, 0, 0, 0, 0}, // L = 56
}};

//! The table's transcription check: every row has exactly L - 1 live entries summing to
//! 73 - B(L), the payload the b1/b2 fields leave behind, and nothing past L - 1.
[[nodiscard]] inline constexpr bool allocationRowsMatchPayload() noexcept {
    for (unsigned L = kImbeMinHarmonics; L <= kImbeMaxHarmonics; ++L) {
        const auto& row = kAllocationRows[L - kImbeMinHarmonics];
        unsigned    sum = 0U;
        for (std::size_t i = 0UZ; i + 1UZ < L; ++i) {
            sum += row[i];
        }
        if (sum != 73U - imbeBandCount(L)) {
            return false;
        }
        for (std::size_t i = L - 1UZ; i < kAllocationStride; ++i) {
            if (row[i] != 0U) {
                return false;
            }
        }
    }
    return true;
}
static_assert(allocationRowsMatchPayload(), "every allocation row carries L-1 entries summing to 73 - B(L)");

//! The 64-level gain quantizer, carried in signed Q5.11.
inline constexpr std::array<std::int16_t, 64UZ> kGainLevelsQ11{-5821, -5518, -5239, -4880, -4549, -4292, -4057, -3760, -3370, -2903, //
    -2583, -2305, -1962, -1601, -1138, -711, -302, 57, 433, 795,                                                                     //
    1132, 1510, 1909, 2333, 2705, 3038, 3376, 3689, 3979, 4339,                                                                      //
    4754, 5129, 5435, 5695, 5991, 6300, 6596, 6969, 7342, 7751,                                                                      //
    8101, 8511, 8835, 9102, 9375, 9698, 10055, 10415, 10762, 11084,                                                                  //
    11403, 11752, 12123, 12468, 12863, 13239, 13615, 13997, 14382, 14770,                                                            //
    15301, 15849, 16640, 17809};

//! The quantizer's transcription check: it is monotone increasing.
[[nodiscard]] inline constexpr bool gainLevelsMonotone() noexcept {
    for (std::size_t i = 1UZ; i < kGainLevelsQ11.size(); ++i) {
        if (kGainLevelsQ11[i] <= kGainLevelsQ11[i - 1UZ]) {
            return false;
        }
    }
    return true;
}
static_assert(gainLevelsMonotone(), "the gain quantizer is a monotone staircase");

//! The gain quantizer in the log2 domain the whole amplitude chain lives in.
inline constexpr std::array<double, 64UZ> kGainLevels = [] {
    std::array<double, 64UZ> t{};
    for (std::size_t i = 0UZ; i < t.size(); ++i) {
        t[i] = static_cast<double>(kGainLevelsQ11[i]) / 2048.0;
    }
    return t;
}();

/**
 * Dequantization step for the five gain words b3..b7, indexed [word][allocated bits], in
 * unsigned Q0.16. The step depends only on the word and its allocation, which collapses the
 * standard's 240-entry per-L table to these 39 values; a zero marks a pairing the allocation
 * table never asks for.
 */
inline constexpr std::array<std::array<std::uint16_t, 11UZ>, 5UZ> kGainStepsQ16{{
    {0, 0, 0, 13206, 8126, 5689, 3047, 1625, 813, 406, 203}, // b3
    {0, 0, 0, 8562, 5269, 3688, 1976, 1054, 527, 263, 0},    // b4
    {0, 0, 9359, 7157, 4404, 3083, 1652, 881, 440, 220, 0},  // b5
    {0, 0, 8077, 6177, 3801, 2661, 1425, 760, 380, 190, 0},  // b6
    {0, 0, 7353, 5623, 3460, 2422, 1298, 692, 346, 173, 0},  // b7
}};

//! The collapse's check: every allocation the table hands b3..b7, over all 48 rows, names a
//! step that exists.
[[nodiscard]] inline constexpr bool gainStepsCoverAllocations() noexcept {
    for (unsigned L = kImbeMinHarmonics; L <= kImbeMaxHarmonics; ++L) {
        for (std::size_t w = 0UZ; w < 5UZ; ++w) {
            if (kGainStepsQ16[w][kAllocationRows[L - kImbeMinHarmonics][w]] == 0U) {
                return false;
            }
        }
    }
    return true;
}
static_assert(gainStepsCoverAllocations(), "every gain-word allocation the table uses has a step");

//! The same steps as reals.
inline constexpr std::array<std::array<double, 11UZ>, 5UZ> kGainSteps = [] {
    std::array<std::array<double, 11UZ>, 5UZ> t{};
    for (std::size_t w = 0UZ; w < 5UZ; ++w) {
        for (std::size_t b = 0UZ; b < 11UZ; ++b) {
            t[w][b] = static_cast<double>(kGainStepsQ16[w][b]) / 65536.0;
        }
    }
    return t;
}();

//! Residual step by DCT coefficient position 1..9, in Q0.16.
inline constexpr std::array<double, 9UZ> kSigma = [] {
    constexpr std::array<std::uint16_t, 9UZ> q{20120, 15794, 13566, 12452, 11731, 11338, 10813, 11141, 11141};
    std::array<double, 9UZ>                  t{};
    for (std::size_t i = 0UZ; i < t.size(); ++i) {
        t[i] = static_cast<double>(q[i]) / 65536.0;
    }
    return t;
}();

//! Residual step by allocated bits 1..10, in Q1.15 — the standard's 1.2, 0.85, 0.65, 0.4,
//! 0.28, 0.15, 0.08, 0.04, 0.02, 0.01.
inline constexpr std::array<double, 10UZ> kDelta = [] {
    constexpr std::array<std::uint16_t, 10UZ> q{39322, 27853, 21299, 13107, 9175, 4915, 2621, 1311, 655, 328};
    std::array<double, 10UZ>                  t{};
    for (std::size_t i = 0UZ; i < t.size(); ++i) {
        t[i] = static_cast<double>(q[i]) / 32768.0;
    }
    return t;
}();

//! The longest block a residual split can produce is ceil(56/6) = 10, so its highest live
//! coefficient position is 9 — exactly what `kSigma` carries.
static_assert(kSigma.size() == (kImbeMaxHarmonics + 5UZ) / 6UZ - 1UZ, "sigma covers every coefficient position a block can reach");

//! The synthesis window, generated rather than tabulated: w(k) = floor((k+1) * 65536/100) in
//! Q15 — the standard's table, produced here by formula rather than transcribed.
inline constexpr std::array<std::uint16_t, 49UZ> kSynthesisWindowQ15 = [] {
    std::array<std::uint16_t, 49UZ> t{};
    for (std::size_t k = 0UZ; k < t.size(); ++k) {
        t[k] = static_cast<std::uint16_t>((k + 1UZ) * 65536UZ / 100UZ);
    }
    return t;
}();

//! Endpoints, so the generating rule is pinned rather than merely restated.
static_assert(kSynthesisWindowQ15[0] == 655U, "w(0) is floor(65536/100)");
static_assert(kSynthesisWindowQ15[48] == 32112U, "w(48) is floor(49 * 65536/100)");

//! The property that makes a transition level-preserving: the ramp rises strictly, and the
//! fade-in and fade-out that meet at every sample of 56..104 — w(k) and w(48 - k) — sum to
//! unity within one Q15 step (exactly at k = 24, where both land on a representable 0.5).
[[nodiscard]] inline constexpr bool synthesisWindowPairsToUnity() noexcept {
    for (std::size_t k = 0UZ; k < kSynthesisWindowQ15.size(); ++k) {
        if (k > 0UZ && kSynthesisWindowQ15[k] <= kSynthesisWindowQ15[k - 1UZ]) {
            return false;
        }
        const unsigned sum = static_cast<unsigned>(kSynthesisWindowQ15[k]) + static_cast<unsigned>(kSynthesisWindowQ15[48UZ - k]);
        if (sum != 32768U && sum != 32767U) {
            return false;
        }
    }
    return true;
}
static_assert(synthesisWindowPairsToUnity(), "the window's fade pair sums to one across the transition");

//! The same window as reals. The fade-in at sample j of 56..104 is w(j - 56) and the fade-out
//! is w(104 - j), so the pair covers the transition and each ramp meets the flat top exactly
//! one step past its end.
inline constexpr std::array<double, 49UZ> kSynthesisWindow = [] {
    std::array<double, 49UZ> t{};
    for (std::size_t k = 0UZ; k < t.size(); ++k) {
        t[k] = static_cast<double>(kSynthesisWindowQ15[k]) / 32768.0;
    }
    return t;
}();

//! Quantizes a band edge down to sixteen fractional bits, then takes the ceiling in units of
//! 1/256. The double rounding is observable at exact-integer edges, which is why it is spelled
//! out rather than folded into one ceiling.
[[nodiscard]] inline int ceilBandEdge(double edge) noexcept {
    const long long floored = static_cast<long long>(edge * 65536.0); // edge > 0 throughout
    return static_cast<int>((floored + 255LL) / 256LL);
}

/**
 * The inverse DCT that serves the gains and every residual block:
 * out[i] = in[0] + 2 * sum over m = 1..M-1 of in[m] * cos(pi * m * (i + 0.5) / M).
 *
 * Written as the direct O(M^2) sum. M is at most ten here — six for the gains, ceil(L/6) for
 * a residual block — so a transform would cost more than it saved.
 */
inline void inverseDct(std::span<const double> in, std::span<double> out) noexcept {
    const std::size_t M = in.size();
    for (std::size_t i = 0UZ; i < M; ++i) {
        double acc = 0.0;
        for (std::size_t m = 1UZ; m < M; ++m) {
            acc += in[m] * std::cos(std::numbers::pi * static_cast<double>(m) * (static_cast<double>(i) + 0.5) / static_cast<double>(M));
        }
        out[i] = in[0] + 2.0 * acc;
    }
}

} // namespace detail

//! The allocation row for L: how many payload bits each of b3..b(L+1) receives.
[[nodiscard]] inline constexpr std::span<const std::uint8_t> imbeAllocationRow(unsigned L) noexcept { return std::span<const std::uint8_t>(detail::kAllocationRows[L - kImbeMinHarmonics].data(), L - 1UZ); }

// ------------------------------------------------------------------------------ the unpacker

/**
 * One codeword's 88 information bits, reorganized into the b-parameter vector: b0 the pitch,
 * b1 the voicing decisions, b2 the gain index, b3..b(L+1) the spectral amplitude data,
 * b(L+2) the synchronization bit.
 */
struct ImbeUnpacked {
    unsigned b0{};   //!< the pitch index, 0..207
    unsigned L{};    //!< harmonics, 9..56
    unsigned B{};    //!< voicing bands, at most twelve
    unsigned b1{};   //!< one voiced/unvoiced bit per band, most significant first
    unsigned b2{};   //!< the six-bit gain quantizer index
    unsigned sync{}; //!< the frame synchronization bit, read by nothing in this decoder and not a parity check

    //! b3..b(L+1) at entries 0..L-2; entries past L-2 are not live.
    std::array<unsigned, 57UZ> b{};

    double w0{}; //!< the fundamental in radians per sample
};

/**
 * The eight parameter words to the b-parameter vector.
 *
 * Returns nullopt when b0 lands in 208..255, which is the frame's signal to conceal; the
 * caller then synthesizes from the previous parameter set.
 *
 * The 75 bits that are neither b0's eight, b2's four direct bits, nor the sync bit form one
 * priority stream, most significant bit first from each word in order. b1 takes B of them
 * from position 39 and b2's middle two follow; what remains — 73 - B bits — distributes over
 * b3..b(L+1) by descending allocation threshold, the rescanning pass implemented below.
 */
[[nodiscard]] inline std::optional<ImbeUnpacked> imbeUnpack(std::span<const std::uint16_t, kImbeParameterWords> u) noexcept {
    const unsigned b0 = ((static_cast<unsigned>(u[0]) >> 6U) << 2U) | ((static_cast<unsigned>(u[7]) >> 1U) & 3U);
    if (b0 > 207U) {
        return std::nullopt;
    }

    // The priority stream: u0's low three bits, then twelve each from u1..u3, eleven each from
    // u4..u6, then u7's bits 6..4. 8 + 4 + 1 + 75 = 88.
    std::array<std::uint8_t, 75UZ> stream{};
    std::size_t                    n = 0UZ;
    for (unsigned k : {2U, 1U, 0U}) {
        stream[n++] = static_cast<std::uint8_t>((static_cast<unsigned>(u[0]) >> k) & 1U);
    }
    for (std::size_t w = 1UZ; w <= 3UZ; ++w) {
        for (unsigned k = 12U; k-- > 0U;) {
            stream[n++] = static_cast<std::uint8_t>((static_cast<unsigned>(u[w]) >> k) & 1U);
        }
    }
    for (std::size_t w = 4UZ; w <= 6UZ; ++w) {
        for (unsigned k = 11U; k-- > 0U;) {
            stream[n++] = static_cast<std::uint8_t>((static_cast<unsigned>(u[w]) >> k) & 1U);
        }
    }
    for (unsigned k : {6U, 5U, 4U}) {
        stream[n++] = static_cast<std::uint8_t>((static_cast<unsigned>(u[7]) >> k) & 1U);
    }

    ImbeUnpacked p;
    p.b0 = b0;
    p.L  = imbeHarmonicCount(b0);
    p.B  = imbeBandCount(p.L);
    p.w0 = imbeFundamental(b0);

    for (std::size_t k = 0UZ; k < p.B; ++k) {
        p.b1 = (p.b1 << 1U) | static_cast<unsigned>(stream[39UZ + k]);
    }
    // b2's bits 5..3 come straight from u0, its bits 2..1 from the stream, its bit 0 from u7.
    p.b2   = (static_cast<unsigned>(u[0]) & 0x38U) | (static_cast<unsigned>(stream[39UZ + p.B]) << 2U) | (static_cast<unsigned>(stream[40UZ + p.B]) << 1U) | ((static_cast<unsigned>(u[7]) >> 3U) & 1U);
    p.sync = static_cast<unsigned>(u[7]) & 1U;

    // What the b1/b2 fields leave: the stream's first 39 bits followed by everything past them.
    std::array<std::uint8_t, 73UZ> payload{};
    std::size_t                    payloadSize = 0UZ;
    for (std::size_t i = 0UZ; i < 39UZ; ++i) {
        payload[payloadSize++] = stream[i];
    }
    for (std::size_t i = 41UZ + p.B; i < stream.size(); ++i) {
        payload[payloadSize++] = stream[i];
    }

    // The rescanning rule: for thresholds descending one at a time from the row's maximum
    // allocation, walk b3..b(L+1) in order and append the next payload bit to every word whose
    // allocation is at least the threshold. The row sums guarantee the payload runs out exactly
    // as the threshold passes 1. Words allocated zero bits stay zero and still hold their place.
    const std::span<const std::uint8_t> alloc     = imbeAllocationRow(p.L);
    int                                 threshold = 0;
    for (std::uint8_t a : alloc) {
        threshold = std::max(threshold, static_cast<int>(a));
    }
    std::size_t pos = 0UZ;
    while (pos < payloadSize && threshold >= 1) {
        for (std::size_t i = 0UZ; i < alloc.size() && pos < payloadSize; ++i) {
            if (static_cast<int>(alloc[i]) >= threshold) {
                p.b[i] = (p.b[i] << 1U) | static_cast<unsigned>(payload[pos]);
                ++pos;
            }
        }
        --threshold;
    }
    return p;
}

// -------------------------------------------------------------------------------- the decoder

/**
 * One voice stream's decoder state.
 *
 * Everything the instance carries between codewords is a member here, and `reset()` rebuilds
 * exactly that set: the predictor's previous log2 amplitudes and its harmonic count; the
 * synthesizer's previous enhanced amplitudes, voicing decisions, harmonic count and
 * fundamental; the 56 phase accumulators; the 105-sample unvoiced tail; the noise generator;
 * and the previous parameter frame concealment falls back to.
 */
struct ImbeDecoder {
    ImbeRng _rng{}; //!< the noise generator, seeded 1 and drawn from in three places per frame

    std::array<double, kImbeMaxHarmonics + 1UZ> _phase{};    //!< phase accumulators, entry l is harmonic l
    ImbeAmplitudes                              _mPrev{};    //!< the predictor's previous unenhanced log2 amplitudes
    unsigned                                    _lPred{30U}; //!< the predictor's previous harmonic count

    ImbeAmplitudes                            _mSynPrev{}; //!< the synthesizer's previous enhanced linear amplitudes
    std::array<bool, kImbeMaxHarmonics + 1UZ> _vuPrev{};   //!< the previous frame's voicing decisions
    unsigned                                  _lSyn{0U};   //!< the synthesizer's previous harmonic count
    double                                    _w0Prev{0.}; //!< the previous fundamental

    std::array<double, kImbeUnvoicedTail> _tail{};   //!< the unvoiced realization carried into the next frame
    ImbeUnpacked                          _params{}; //!< the parameter set a concealed frame reuses

    ImbeDecoder() noexcept { reset(); }

    /**
     * Rebuilds the construction state exactly.
     *
     * The 56 phase accumulators come from the first 56 draws of a fresh generator, one per
     * harmonic in order, at r * pi each — so the 57th draw is the first frame's first draw of
     * the reset generator. The concealment predecessor is primed as the b0 = 0 frame: nine
     * harmonics, three bands, all of them unvoiced, all-zero residual data.
     */
    void reset() noexcept {
        _rng = ImbeRng{};
        _phase.fill(0.0);
        for (std::size_t l = 1UZ; l <= kImbeMaxHarmonics; ++l) {
            _phase[l] = std::numbers::pi * _rng.draw();
        }
        _mPrev.fill(0.0);
        _lPred = 30U;
        _mSynPrev.fill(0.0);
        _vuPrev.fill(false);
        _lSyn   = 0U;
        _w0Prev = 0.0;
        _tail.fill(0.0);

        _params    = ImbeUnpacked{};
        _params.b0 = 0U;
        _params.L  = imbeHarmonicCount(0U);
        _params.B  = imbeBandCount(_params.L);
        _params.w0 = imbeFundamental(0U);
    }

    /**
     * The spectral amplitude reconstruction: gain dequantization, the six-block inverse DCT of
     * the residuals, and the blend with the resampled previous frame.
     *
     * Returns the linear amplitudes m[1..L]; entry 0 and everything past L are zero. Advances
     * the predictor state exactly as `decode()` does; the noise generator, the phases and the
     * tail are untouched here.
     */
    [[nodiscard]] ImbeAmplitudes amplitudes(const ImbeUnpacked& p) noexcept {
        const unsigned                      L     = p.L;
        const std::span<const std::uint8_t> alloc = imbeAllocationRow(L);

        // Six gain values: the quantizer at index b2, then b3..b7 by the uniform rule
        // value = step * (q - 2^(bits-1) + 0.5).
        std::array<double, 6UZ> gains{};
        gains[0] = detail::kGainLevels[p.b2];
        for (std::size_t i = 1UZ; i < 6UZ; ++i) {
            const unsigned bits = alloc[i - 1UZ];
            gains[i]            = detail::kGainSteps[i - 1UZ][bits] * (static_cast<double>(p.b[i - 1UZ]) - static_cast<double>(1U << (bits - 1U)) + 0.5);
        }
        std::array<double, 6UZ> blockGain{};
        detail::inverseDct(gains, blockGain);

        // The L residuals split into six consecutive blocks of floor(L/6) or ceil(L/6), the
        // longer ones last: exactly L mod 6 blocks of ceil(L/6) at the end.
        const unsigned            q = L / 6U;
        const unsigned            r = L % 6U;
        std::array<unsigned, 6UZ> lengths{};
        for (std::size_t i = 0UZ; i < 6UZ; ++i) {
            lengths[i] = (i < 6UZ - r) ? q : q + 1U;
        }

        ImbeAmplitudes T{}; // the log2 residuals, 1-indexed
        std::size_t    k     = 5UZ;
        std::size_t    outAt = 1UZ;
        for (std::size_t blk = 0UZ; blk < 6UZ; ++blk) {
            const std::size_t        len = lengths[blk];
            std::array<double, 10UZ> c{};
            c[0] = blockGain[blk];
            for (std::size_t j = 1UZ; j < len; ++j) {
                const unsigned bits = alloc[k];
                if (bits != 0U) {
                    c[j] = detail::kSigma[j - 1UZ] * detail::kDelta[bits - 1UZ] * (static_cast<double>(p.b[k]) - static_cast<double>(1U << (bits - 1U)) + 0.5);
                }
                ++k; // a zero-allocation word contributes a zero coefficient and still consumes its position
            }
            detail::inverseDct(std::span<const double>(c.data(), len), std::span<double>(T.data() + outAt, len));
            outAt += len;
        }

        // The previous frame's log amplitudes resample onto the new harmonic grid and blend
        // with the residuals; the mean removal keeps predicted log energy level.
        const unsigned                              Lp = _lPred;
        std::array<double, kImbeMaxHarmonics + 3UZ> prev{};
        for (std::size_t i = 0UZ; i <= Lp; ++i) {
            prev[i] = _mPrev[i];
        }
        for (std::size_t i = Lp + 1UZ; i < prev.size(); ++i) {
            prev[i] = _mPrev[Lp]; // Mprev extends past Lprev by repeating its last entry
        }

        const double rho = (L <= 15U) ? 0.4 : ((L <= 24U) ? 0.03 * static_cast<double>(L) - 0.05 : 0.7);

        ImbeAmplitudes interp{};
        for (std::size_t l = 1UZ; l <= L; ++l) {
            const double      pos = static_cast<double>(l * Lp) / static_cast<double>(L);
            const std::size_t i0  = static_cast<std::size_t>(pos);
            const double      f   = pos - static_cast<double>(i0);
            interp[l]             = (1.0 - f) * prev[i0] + f * prev[i0 + 1UZ];
        }
        double mean = 0.0;
        for (std::size_t l = 1UZ; l <= L; ++l) {
            mean += interp[l];
        }
        mean *= rho / static_cast<double>(L);

        _mPrev.fill(0.0);
        ImbeAmplitudes m{};
        for (std::size_t l = 1UZ; l <= L; ++l) {
            const double M = T[l] + rho * interp[l] - mean;
            _mPrev[l]      = M;
            m[l]           = std::exp2(M);
        }
        _lPred = L;
        return m;
    }

    /**
     * Spectral amplitude enhancement, in place over m[1..L].
     *
     * Two moments of the decoded spectrum sharpen the formants: R0 = sum of m[l]^2 and
     * R1 = sum of m[l]^2 * cos(w0*l). Every harmonic above the lowest eighth (8*l > L) with a
     * nonzero amplitude scales by W, clamped to 0.5..1.2; the lowest eighth passes unchanged.
     * If the enhanced energy exceeds R0 the whole vector rescales — enhancement never adds
     * energy.
     *
     * The units in W are load-bearing. The cosines take w0 in radians per sample, while the
     * denominator's frequency is the fundamental as a fraction of the sample rate, w0/2pi;
     * reading it as radians shrinks W by (2pi)^(1/4), about 2 dB of formant level. The 0.9898
     * is the standard's 0.96 folded into the fourth root.
     *
     * Carries no state, so it moves the decoder no further than `decode()` does at this point.
     */
    void enhance(ImbeAmplitudes& m, unsigned L, double w0) const noexcept {
        double R0 = 0.0;
        for (std::size_t l = 1UZ; l <= L; ++l) {
            R0 += m[l] * m[l];
        }
        if (R0 <= 0.0) {
            return;
        }
        ImbeAmplitudes cosw{};
        double         R1 = 0.0;
        for (std::size_t l = 1UZ; l <= L; ++l) {
            cosw[l] = std::cos(w0 * static_cast<double>(l));
            R1 += m[l] * m[l] * cosw[l];
        }

        const double den = (w0 / (2.0 * std::numbers::pi)) * R0 * (R0 * R0 - R1 * R1);
        if (den <= 0.0) {
            return;
        }
        for (std::size_t l = 1UZ; l <= L; ++l) {
            if (8UZ * l <= L || m[l] == 0.0) {
                continue;
            }
            const double num = m[l] * m[l] * (0.25 * (R0 * R0 + R1 * R1) - 0.5 * R0 * R1 * cosw[l]);
            const double w   = 0.9898 * std::pow(num / den, 0.25);
            m[l] *= std::clamp(w, 0.5, 1.2);
        }

        double e = 0.0;
        for (std::size_t l = 1UZ; l <= L; ++l) {
            e += m[l] * m[l];
        }
        if (e > R0) {
            const double scale = std::sqrt(R0 / e);
            for (std::size_t l = 1UZ; l <= L; ++l) {
                m[l] *= scale;
            }
        }
    }

    /**
     * One codeword's eight parameter words to 160 samples.
     *
     * Each output sample is the voiced and unvoiced components summed, saturated to the signed
     * 16-bit rails and quantized to the int16 grid by truncation toward zero, then divided by
     * 32768 — so every value is n/32768 for an integer n, full scale plus or minus one.
     */
    void decode(std::span<const std::uint16_t, kImbeParameterWords> u, std::span<float, kImbeSamplesPerFrame> out) noexcept {
        const std::optional<ImbeUnpacked> unpacked = imbeUnpack(u);
        if (unpacked.has_value()) {
            _params = *unpacked;
        }
        // Concealment retains the parameter set and lets synthesis proceed with it: the
        // prediction re-runs against the newest predictor state and the synthesizer keeps
        // advancing phases and drawing noise, so a repeated frame is not a repeated output.
        const ImbeUnpacked& p = _params;

        // Band k covers harmonics 3k+1, 3k+2, 3k+3; the last band covers every harmonic past
        // 3(B-1).
        std::array<bool, kImbeMaxHarmonics + 1UZ> vu{};
        unsigned                                  band        = 0U;
        unsigned                                  voicedCount = 0U;
        for (unsigned l = 1U; l <= p.L; ++l) {
            vu[l] = ((p.b1 >> (p.B - 1U - band)) & 1U) != 0U;
            voicedCount += vu[l] ? 1U : 0U;
            if (l % 3U == 0U && band < p.B - 1U) {
                ++band;
            }
        }
        const unsigned Luv = p.L - voicedCount;

        ImbeAmplitudes m = amplitudes(p);
        enhance(m, p.L, p.w0);

        std::array<double, kImbeSamplesPerFrame> voiced{};
        std::array<double, kImbeSamplesPerFrame> unvoiced{};
        synthesizeVoiced(p, m, vu, Luv, voiced);
        synthesizeUnvoiced(p, m, vu, unvoiced);

        for (std::size_t j = 0UZ; j < kImbeSamplesPerFrame; ++j) {
            const double s = std::trunc(std::clamp(voiced[j] + unvoiced[j], -32768.0, 32767.0));
            out[j]         = static_cast<float>(s / 32768.0);
        }
    }

private:
    /**
     * The voiced synthesizer.
     *
     * Every one of the 56 accumulators advances first, by 160 * l * (w0prev + w0)/2 — the
     * average of the two fundamentals — modulo 2*pi. The stage-2 draws then perturb the upper
     * harmonics, one draw per l in 1..Lmax with l - 1 > floor(Lmax/4) whether that harmonic is
     * voiced or not. Lmax is max(L, the synthesizer's previous count) — never the predictor's,
     * whose initial 30 would change the first frame's draw count and every draw after it.
     *
     * `_phase[l]` afterwards is the harmonic's phase at the end of the frame, perturbation
     * included, and every branch anchors to it.
     */
    void synthesizeVoiced(const ImbeUnpacked& p, const ImbeAmplitudes& m, const std::array<bool, kImbeMaxHarmonics + 1UZ>& vu, unsigned Luv, std::span<double, kImbeSamplesPerFrame> out) noexcept {
        const unsigned L    = p.L;
        const double   w0   = p.w0;
        const unsigned Lmax = std::max(L, _lSyn);

        const std::array<double, kImbeMaxHarmonics + 1UZ> phiStart = _phase;

        const double adv = 160.0 * (_w0Prev + w0) / 2.0;
        for (std::size_t l = 1UZ; l <= kImbeMaxHarmonics; ++l) {
            _phase[l] = std::fmod(_phase[l] + static_cast<double>(l) * adv, 2.0 * std::numbers::pi);
        }

        std::array<double, kImbeMaxHarmonics + 1UZ> delta{};
        const unsigned                              quarter = Lmax >> 2U;
        for (unsigned l = 1U; l <= Lmax; ++l) {
            if (l - 1U > quarter) {
                const double r = _rng.draw();
                delta[l]       = (Luv == L) ? std::numbers::pi * r : (std::numbers::pi / 2.0) * r * static_cast<double>(Luv) / static_cast<double>(L);
                _phase[l] += delta[l];
            }
        }

        // The continuous branch needs a low harmonic and a pitch that barely moved; the
        // threshold is on the current fundamental.
        const bool bigStep = std::abs(w0 - _w0Prev) >= 0.1 * w0;

        for (unsigned l = 1U; l <= Lmax; ++l) {
            const bool newV = vu[l];
            const bool oldV = _vuPrev[l];
            if (!newV && !oldV) {
                continue; // the unvoiced synthesizer owns it
            }
            const double amp    = m[l];
            const double ampOld = _mSynPrev[l];

            // The new sinusoid's phase at sample j is phi[l] + (j - 160) * l * w0, the old
            // one's is phiprev[l] + j * l * w0prev.
            const auto newCos = [&](std::size_t j) noexcept { return std::cos(_phase[l] + (static_cast<double>(j) - 160.0) * static_cast<double>(l) * w0); };
            const auto oldCos = [&](std::size_t j) noexcept { return std::cos(phiStart[l] + static_cast<double>(j) * static_cast<double>(l) * _w0Prev); };

            const auto fadeInNew = [&]() noexcept {
                for (std::size_t j = 56UZ; j < 105UZ; ++j) {
                    out[j] += 2.0 * amp * detail::kSynthesisWindow[j - 56UZ] * newCos(j);
                }
                for (std::size_t j = 105UZ; j < kImbeSamplesPerFrame; ++j) {
                    out[j] += 2.0 * amp * newCos(j);
                }
            };
            const auto fadeOutOld = [&]() noexcept {
                for (std::size_t j = 0UZ; j < 56UZ; ++j) {
                    out[j] += 2.0 * ampOld * oldCos(j);
                }
                for (std::size_t j = 56UZ; j < 105UZ; ++j) {
                    out[j] += 2.0 * ampOld * detail::kSynthesisWindow[104UZ - j] * oldCos(j);
                }
            };

            if (newV && !oldV) {
                fadeInNew(); // nothing before sample 56
            } else if (oldV && !newV) {
                fadeOutOld();
            } else if (l >= 8U || bigStep) {
                fadeOutOld();
                fadeInNew();
            } else {
                // One continuous sinusoid across all 160 samples: the amplitude is exactly the
                // previous one at j = 0 and reaches the new one a sample past the frame, so
                // consecutive frames join without a step, and the phase lands exactly on the
                // stored phi[l] at j = 160 with the perturbation riding the j*delta/160 term —
                // already inside phi[l], so nothing adds delta twice.
                for (std::size_t j = 0UZ; j < kImbeSamplesPerFrame; ++j) {
                    const double jd = static_cast<double>(j);
                    const double a  = ampOld + (amp - ampOld) * jd / 160.0;
                    const double ph = phiStart[l] + jd * static_cast<double>(l) * _w0Prev + jd * jd * static_cast<double>(l) * (w0 - _w0Prev) / 320.0 + jd * delta[l] / 160.0;
                    out[j] += 2.0 * a * std::cos(ph);
                }
            }
        }

        _vuPrev.fill(false);
        for (std::size_t l = 1UZ; l <= L; ++l) {
            _vuPrev[l]   = vu[l];
            _mSynPrev[l] = m[l];
        }
        _lSyn   = L;
        _w0Prev = w0;
    }

    /**
     * The unvoiced synthesizer.
     *
     * Harmonic l's band spans (l - 0.5)*f0 to (l + 0.5)*f0 in cycles per sample, and its bins
     * run from ceil16(low) up to but excluding ceil16(high). Each bin of an unvoiced band
     * takes two stage-3 draws, real part then imaginary, bins ascending, harmonics in order,
     * voiced harmonics drawing nothing. The realization is
     * the direct 256-point sum over the active bins — there are at most 128 of them, and the
     * conjugate half is folded into the real-valued form, so a transform would cost more than
     * it saved.
     *
     * The frame then lays that realization over a 105-sample memory: samples 0..55 are the
     * previous tail alone, 56..104 crossfade it with the new realization, 105..159 are the new
     * realization alone. Output sample j takes transform index j - 32; the tail saved for the
     * next frame is transform indices 128..232.
     */
    void synthesizeUnvoiced(const ImbeUnpacked& p, const ImbeAmplitudes& m, const std::array<bool, kImbeMaxHarmonics + 1UZ>& vu, std::span<double, kImbeSamplesPerFrame> out) noexcept {
        const unsigned L  = p.L;
        const double   f0 = p.w0 / (2.0 * std::numbers::pi);

        // At most one entry per positive-frequency bin: the bands are contiguous and never
        // reach bin 128.
        std::array<int, kImbeNoiseBins / 2UZ>    bins{};
        std::array<double, kImbeNoiseBins / 2UZ> real{};
        std::array<double, kImbeNoiseBins / 2UZ> imag{};
        std::size_t                              count = 0UZ;

        for (unsigned l = 1U; l <= L; ++l) {
            const int low  = detail::ceilBandEdge((static_cast<double>(l) - 0.5) * f0);
            const int high = detail::ceilBandEdge((static_cast<double>(l) + 0.5) * f0);
            if (vu[l]) {
                continue; // a voiced band's bins stay zero and no draw happens
            }
            for (int k = low; k < high && count < bins.size(); ++k) {
                const double r1 = _rng.draw();
                const double r2 = _rng.draw();
                bins[count]     = k;
                real[count]     = m[l] * r1;
                imag[count]     = m[l] * r2;
                ++count;
            }
        }

        std::array<double, kImbeNoiseBins> x{};
        if (count != 0UZ) {
            constexpr double twoPiOver256 = 2.0 * std::numbers::pi / static_cast<double>(kImbeNoiseBins);
            for (std::size_t n = 0UZ; n < kImbeNoiseBins; ++n) {
                double cosSum = 0.0;
                double sinSum = 0.0;
                for (std::size_t e = 0UZ; e < count; ++e) {
                    const double theta = twoPiOver256 * static_cast<double>(bins[e] * static_cast<int>(n));
                    cosSum += real[e] * std::cos(theta);
                    sinSum += imag[e] * std::sin(theta);
                }
                x[n] = 2.0 * (cosSum + sinSum);
            }
        }

        for (std::size_t j = 0UZ; j < 56UZ; ++j) {
            out[j] = _tail[j];
        }
        for (std::size_t j = 56UZ; j < 105UZ; ++j) {
            out[j] = _tail[j] * detail::kSynthesisWindow[104UZ - j] + x[j - 32UZ] * detail::kSynthesisWindow[j - 56UZ];
        }
        for (std::size_t j = 105UZ; j < kImbeSamplesPerFrame; ++j) {
            out[j] = x[j - 32UZ];
        }
        for (std::size_t i = 0UZ; i < kImbeUnvoicedTail; ++i) {
            _tail[i] = x[128UZ + i];
        }
    }
};

} // namespace gr::vocoder

#endif // GNURADIO_ALGORITHM_VOCODER_IMBE_HPP
