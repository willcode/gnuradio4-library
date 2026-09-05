#include <boost/ut.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <gnuradio-4.0/algorithm/vocoder/Imbe.hpp>

/*
 * The IMBE 7200x4400 decoder is specified in real arithmetic, so these tests pin what the
 * specification pins and refuse to pin what it deliberately leaves free.
 *
 * What is pinned exactly: the bit fields (the unpack probe exercises every field boundary, and
 * the three on-air parameter sets are real transmitted voice, where a wrong boundary cannot
 * hide), the integer rules (the L staircase), the noise generator and its consumption order
 * (twelve outputs plus the two anchors that catch a decoder drawing the wrong number of times
 * at construction), and the amplitude chain end to end, including the prediction re-blend that
 * only shows up on the second codeword.
 *
 * What is pinned structurally rather than sample for sample: synthesis. Two fresh decoders
 * agree and a reset one matches a fresh one, because the noise generator is integer and its
 * consumption order is codec state; a concealed frame is not a repeat of the previous output,
 * because phases, predictor, tail and seed all keep advancing through it; and every output
 * sample lands on the int16 grid, because the output is PCM. Golden PCM is deliberately absent:
 * last-ulp arithmetic floats across platforms, and sample-exact equality with any particular
 * realization is not a conformance criterion.
 */
namespace {

using gr::vocoder::ImbeAmplitudes;
using gr::vocoder::ImbeDecoder;
using gr::vocoder::imbeHarmonicCount;
using gr::vocoder::ImbeRng;
using gr::vocoder::imbeUnpack;
using gr::vocoder::kImbeParameterWords;
using gr::vocoder::kImbeSamplesPerFrame;

using Codeword = std::array<std::uint16_t, kImbeParameterWords>;

//! Three on-air parameter sets: real transmitted voice, a fully unvoiced stretch.
constexpr std::array<Codeword, 3UZ> kOnAir{{
    {{398, 1333, 3825, 441, 0, 1733, 1011, 114}},
    {{390, 952, 3781, 3808, 57, 1090, 1301, 19}},
    {{389, 333, 713, 2232, 54, 1523, 133, 66}},
}};

//! Eighteen consecutive voice codewords a real P25 transmitter sent -- two whole voice
//! frames, 360 ms of speech, in the order they went out. Consecutive matters: the codec
//! decodes each set against the one before it, so a fixture that jumped about would ask it
//! to do something it is never asked to do on the air.
//!
//! Captured from a real transmission by a 483.125 MHz P25 repeater.
constexpr std::array<Codeword, 18UZ> kAirParameters{{
    {{402, 301, 1264, 3494, 50, 58, 508, 26}},
    {{400, 762, 3246, 1855, 58, 1816, 1065, 91}},
    {{410, 698, 1037, 1925, 33, 822, 2044, 114}},
    {{408, 639, 3900, 3520, 27, 2031, 41, 115}},
    {{408, 2913, 2294, 2365, 46, 1953, 90, 50}},
    {{415, 1924, 864, 3648, 34, 462, 239, 27}},
    {{2343, 1712, 1267, 3079, 0, 919, 155, 42}},
    {{399, 3968, 1600, 2634, 60, 35, 1722, 19}},
    {{386, 2365, 921, 2456, 53, 754, 2009, 114}},
    {{391, 523, 673, 2472, 63, 1516, 488, 83}},
    {{388, 619, 3760, 3706, 57, 684, 1899, 106}},
    {{386, 2111, 2045, 2800, 55, 1154, 393, 35}},
    {{389, 476, 713, 3296, 40, 1504, 1693, 122}},
    {{390, 809, 1685, 2416, 35, 563, 618, 59}},
    {{388, 111, 4080, 2538, 47, 553, 390, 74}},
    {{389, 1349, 2648, 2698, 45, 876, 1812, 123}},
    {{2703, 1926, 3900, 1202, 0, 877, 463, 52}},
    {{399, 1936, 1455, 2808, 6, 630, 253, 11}},
}};

//! Decodes every codeword in order on one decoder and returns the whole stream.
[[nodiscard]] std::vector<float> decodeAll(ImbeDecoder& decoder, std::span<const Codeword> codewords) {
    std::vector<float> pcm(codewords.size() * kImbeSamplesPerFrame, 0.0F);
    for (std::size_t i = 0UZ; i < codewords.size(); ++i) {
        decoder.decode(codewords[i], std::span<float, kImbeSamplesPerFrame>(pcm.data() + i * kImbeSamplesPerFrame, kImbeSamplesPerFrame));
    }
    return pcm;
}

//! Relative to 5e-6, floored at an absolute 5e-6 so amplitudes below unity are held as tightly.
[[nodiscard]] double amplitudeTolerance(double expected) { return 5e-6 * std::max(1.0, std::abs(expected)); }

} // namespace

const boost::ut::suite<"imbe"> imbeTests = [] {
    using namespace boost::ut;

    "the noise generator, and the consumption order its anchors catch"_test = [] {
        // Park-Miller from seed 1, each draw the new seed's low sixteen bits read signed.
        constexpr std::array<int, 12UZ> expected{16807, 15089, -21287, 3114, -18558, -9528, -28968, 2558, 12099, 1101, -26472, 15445};
        ImbeRng                         rng;
        for (std::size_t i = 0UZ; i < expected.size(); ++i) {
            expect(eq(static_cast<int>(rng.draw() * 32768.0), expected[i])) << "draw" << i;
        }

        // Construction spends exactly 56 draws on the phase accumulators, so the 57th is the
        // first frame's first draw. A decoder that drew one too few or one too many at
        // construction would still pass the twelve above and fail here.
        ImbeRng anchored;
        double  last = 0.0;
        for (std::size_t i = 0UZ; i < 56UZ; ++i) {
            last = anchored.draw();
        }
        expect(eq(static_cast<int>(last * 32768.0), 7074)) << "the 56th draw, the last of construction";
        expect(eq(static_cast<int>(anchored.draw() * 32768.0), 13212)) << "the 57th draw, the first frame's first";
    };

    "the harmonic count is an exact staircase in the pitch index"_test = [] {
        expect(eq(imbeHarmonicCount(0U), 9U)) << "L starts at 9";
        expect(eq(imbeHarmonicCount(20U), 13U));
        expect(eq(imbeHarmonicCount(64U), 24U));
        expect(eq(imbeHarmonicCount(120U), 37U));
        expect(eq(imbeHarmonicCount(128U), 38U));
        expect(eq(imbeHarmonicCount(172U), 49U));
        expect(eq(imbeHarmonicCount(204U), 56U));
        expect(eq(imbeHarmonicCount(207U), 56U)) << "and stops at 56, the last four indices sharing it";

        // The rule never leaves its stated range, and never steps backwards.
        unsigned previous = 0U;
        for (unsigned b0 = 0U; b0 <= 207U; ++b0) {
            const unsigned L = imbeHarmonicCount(b0);
            expect(ge(L, 9U) and le(L, 56U)) << "L in range at b0" << b0;
            expect(ge(L, previous)) << "the staircase never descends, at b0" << b0;
            previous = L;
        }
    };

    "the unpack probe, which exercises every field boundary"_test = [] {
        constexpr Codeword probe{0x801, 0xAAA, 0x555, 0xFFF, 0x401, 0x2AA, 0x555, 0x41};
        const auto         p = imbeUnpack(probe);
        expect(fatal(p.has_value())) << "b0 is inside the legal range, so this frame decodes";
        expect(eq(p->b0, 128U));
        expect(eq(p->L, 38U));
        expect(eq(p->B, 12U)) << "the band count is at its cap";
        expect(eq(p->b1, 2050U));
        expect(eq(p->b2, 4U));
        expect(eq(p->sync, 1U));

        constexpr std::array<unsigned, 37UZ> expected{1, 7, 5, 3, 1, 7, 1, 1, 3, 1, 1, 3, 1, 3, 0, 5, 2, 1, 0, 1, 0, 1, 0, 1, 0, 3, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0};
        for (std::size_t i = 0UZ; i < expected.size(); ++i) {
            expect(eq(p->b[i], expected[i])) << "b" << (i + 3UZ);
        }
    };

    "the three on-air parameter sets, where a wrong field boundary cannot hide"_test = [] {
        constexpr std::array<unsigned, 3UZ>                   expectedB2{8U, 6U, 6U};
        constexpr std::array<unsigned, 3UZ>                   expectedSync{0U, 1U, 0U};
        constexpr std::array<std::array<unsigned, 13UZ>, 3UZ> expectedWords{{
            {81, 47, 31, 41, 31, 87, 52, 16, 9, 7, 1, 3, 1},
            {86, 27, 44, 36, 36, 83, 56, 17, 8, 9, 6, 8, 3},
            {64, 26, 29, 34, 22, 46, 70, 19, 8, 9, 3, 6, 4},
        }};

        for (std::size_t c = 0UZ; c < kOnAir.size(); ++c) {
            const auto p = imbeUnpack(kOnAir[c]);
            expect(fatal(p.has_value())) << "codeword" << c;
            expect(eq(p->b0, 25U)) << "codeword" << c;
            expect(eq(p->L, 14U)) << "codeword" << c;
            expect(eq(p->B, 5U)) << "codeword" << c;
            expect(eq(p->b1, 0U)) << "a fully unvoiced stretch of speech, codeword" << c;
            expect(eq(p->b2, expectedB2[c])) << "codeword" << c;
            expect(eq(p->sync, expectedSync[c])) << "codeword" << c;
            for (std::size_t i = 0UZ; i < expectedWords[c].size(); ++i) {
                expect(eq(p->b[i], expectedWords[c][i])) << "codeword" << c << "b" << (i + 3UZ);
            }
        }
    };

    "the amplitude chain, from a fresh decoder through the prediction re-blend"_test = [] {
        constexpr std::array<double, 14UZ> beforeEnhancement{2.00271, 0.646059, 0.284692, 0.495247, 0.25626, 0.235552, 0.215952, 0.150495, 0.108661, 0.264091, 0.125922, 0.144231, 0.632193, 0.54364};
        constexpr std::array<double, 14UZ> afterEnhancement{2.00271, 0.439283, 0.145268, 0.370217, 0.150381, 0.142503, 0.132896, 0.0813204, 0.0543305, 0.204156, 0.0691551, 0.0867122, 0.758632, 0.652368};
        constexpr std::array<double, 14UZ> secondEnhanced{2.31675, 0.44989, 0.118631, 0.325621, 0.314825, 0.222286, 0.288857, 0.204981, 0.189004, 0.070667, 0.113645, 0.0313967, 0.0886799, 0.0575314};

        ImbeDecoder decoder;

        const auto first = imbeUnpack(kOnAir[0]);
        expect(fatal(first.has_value()));
        ImbeAmplitudes m = decoder.amplitudes(*first);
        for (std::size_t l = 1UZ; l <= 14UZ; ++l) {
            expect(approx(m[l], beforeEnhancement[l - 1UZ], amplitudeTolerance(beforeEnhancement[l - 1UZ]))) << "m[" << l << "] before enhancement";
        }

        const double lowestEighth = m[1];
        decoder.enhance(m, first->L, first->w0);
        for (std::size_t l = 1UZ; l <= 14UZ; ++l) {
            expect(approx(m[l], afterEnhancement[l - 1UZ], amplitudeTolerance(afterEnhancement[l - 1UZ]))) << "m[" << l << "] after enhancement";
        }
        expect(eq(m[1], lowestEighth)) << "m[1] is in the lowest eighth (8*1 <= 14) and passes through untouched";

        // The second set fed next, the predictor now carrying the first's log amplitudes: this
        // is the only place the prediction re-blend shows itself.
        const auto second = imbeUnpack(kOnAir[1]);
        expect(fatal(second.has_value()));
        ImbeAmplitudes m2 = decoder.amplitudes(*second);
        decoder.enhance(m2, second->L, second->w0);
        for (std::size_t l = 1UZ; l <= 14UZ; ++l) {
            expect(approx(m2[l], secondEnhanced[l - 1UZ], amplitudeTolerance(secondEnhanced[l - 1UZ]))) << "second codeword m[" << l << "]";
        }
    };

    "a decoder is deterministic, and reset is exactly construction"_test = [] {
        ImbeDecoder              a;
        ImbeDecoder              b;
        const std::vector<float> first  = decodeAll(a, kOnAir);
        const std::vector<float> second = decodeAll(b, kOnAir);
        expect(fatal(eq(first.size(), 480UZ))) << "three codewords of 160 samples";
        expect(that % (first == second)) << "two fresh instances fed the same codewords agree sample for sample";

        a.reset();
        const std::vector<float> afterReset = decodeAll(a, kOnAir);
        expect(that % (first == afterReset)) << "a reset instance matches a fresh one";
    };

    "a concealed frame advances synthesis rather than repeating output"_test = [] {
        // b0 = ((0xFFF >> 6) << 2) | ((7 >> 1) & 3) = 255, inside the concealment range.
        constexpr Codeword concealed{0xFFF, 0, 0, 0, 0, 0, 0, 7};
        expect(not imbeUnpack(concealed).has_value()) << "b0 of 208..255 has no parameter set of its own";

        ImbeDecoder                             decoder;
        std::array<float, kImbeSamplesPerFrame> previous{};
        std::array<float, kImbeSamplesPerFrame> repeated{};
        decoder.decode(kOnAir[0], previous);
        decoder.decode(concealed, repeated);

        expect(that % (previous != repeated)) << "the repeat is of parameters, not of output: phases, predictor, tail and seed have all moved";
    };

    "a fresh decoder's first frame is confined the way an unvoiced start must be"_test = [] {
        // The first on-air set is fully unvoiced (b1 = 0) and the unvoiced tail starts zeroed,
        // so samples 0..55 -- which are the tail alone -- can only be silence, while the new
        // realization arrives from sample 56 on.
        ImbeDecoder                             decoder;
        std::array<float, kImbeSamplesPerFrame> frame{};
        decoder.decode(kOnAir[0], frame);

        for (std::size_t j = 0UZ; j < 56UZ; ++j) {
            expect(eq(frame[j], 0.0F)) << "sample" << j << "is the zeroed tail alone";
        }
        std::size_t nonzero = 0UZ;
        for (std::size_t j = 56UZ; j < kImbeSamplesPerFrame; ++j) {
            nonzero += (frame[j] != 0.0F) ? 1UZ : 0UZ;
        }
        expect(gt(nonzero, 0UZ)) << "and the new realization does arrive";
    };

    "every output sample is PCM: an int16-grid integer over 32768"_test = [] {
        // Quantization to the grid is part of the output contract, not an artifact of the
        // arithmetic: the sum saturates at the rails and is then truncated toward zero.
        ImbeDecoder              decoder;
        const std::vector<float> pcm = decodeAll(decoder, kOnAir);

        std::size_t offGrid    = 0UZ;
        std::size_t outOfRange = 0UZ;
        for (float sample : pcm) {
            const double scaled = static_cast<double>(sample) * 32768.0;
            offGrid += (scaled != std::trunc(scaled)) ? 1UZ : 0UZ;
            outOfRange += (scaled < -32768.0 || scaled > 32767.0) ? 1UZ : 0UZ;
        }
        expect(eq(offGrid, 0UZ)) << "every sample times 32768 is an integer";
        expect(eq(outOfRange, 0UZ)) << "and lies in -32768..32767";
    };

    "the eighteen-codeword air fixture decodes to bounded, gridded, reproducible PCM"_test = [] {
        ImbeDecoder              decoder;
        const std::vector<float> pcm = decodeAll(decoder, kAirParameters);
        expect(fatal(eq(pcm.size(), 2880UZ))) << "eighteen codewords of 160 samples";

        // The output is PCM: every sample is an integer of the int16 grid, divided by 32768.
        std::size_t nonzero    = 0UZ;
        std::size_t offGrid    = 0UZ;
        std::size_t outOfRange = 0UZ;
        for (float sample : pcm) {
            const double scaled = static_cast<double>(sample) * 32768.0;
            offGrid += (scaled != std::trunc(scaled)) ? 1UZ : 0UZ;
            outOfRange += (scaled < -32768.0 || scaled > 32767.0) ? 1UZ : 0UZ;
            nonzero += (sample != 0.0F) ? 1UZ : 0UZ;
        }
        expect(eq(offGrid, 0UZ)) << "every sample lands on the int16 grid";
        expect(eq(outOfRange, 0UZ)) << "every sample is inside the int16 rails";
        expect(gt(nonzero, 0UZ)) << "real parameters produce real audio, not silence";

        ImbeDecoder              other;
        const std::vector<float> again = decodeAll(other, kAirParameters);
        expect(that % (pcm == again)) << "and the whole stream is byte-identical across two fresh instances";
    };
};

int main() { /* tests run through the boost::ut suite registration above */ }
