#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <format>
#include <numbers>
#include <print>
#include <set>
#include <span>
#include <vector>

#include <gnuradio-4.0/algorithm/fourier/fft.hpp>
#include <gnuradio-4.0/algorithm/lte/SyncSignals.hpp>
#include <gnuradio-4.0/algorithm/rng/Xoshiro256pp.hpp>

using namespace boost::ut;
using gr::lte::CyclicPrefix;
using gr::lte::DuplexMode;
using gr::lte::FrameGeometry;

namespace {

using Complex = std::complex<float>;

/// The cyclic cross-correlation peak of two length-62 sequences, normalized so a sequence against itself is 1.
[[nodiscard]] double crossCorrelationPeak(std::span<const std::complex<double>> a, std::span<const std::complex<double>> b) {
    double best = 0.;
    for (std::size_t shift = 0UZ; shift < gr::lte::kSignalLength; ++shift) {
        std::complex<double> sum{};
        for (std::size_t n = 0UZ; n < gr::lte::kSignalLength; ++n) {
            sum += a[n] * std::conj(b[(n + shift) % gr::lte::kSignalLength]);
        }
        best = std::max(best, std::abs(sum) / static_cast<double>(gr::lte::kSignalLength));
    }
    return best;
}

/**
 * @brief One clean 10 ms radio frame at 1.92 MS/s, on the central six resource blocks.
 *
 * Every OFDM symbol carries seeded random QPSK of unit power on all 72 subcarriers, except that the two
 * synchronization symbols carry their sequences on the central 62 with the outer ten still QPSK. Cyclic prefixes
 * follow the geometry under test, so the two structures and the two prefix types each produce their own frame.
 * This is the clean vector the kernels are held to; a channel is the consuming block's QA.
 */
struct Frame {
    std::vector<Complex>       samples;    ///< 19200 samples, the frame's own first sample at index 0
    std::vector<std::size_t>   pss;        ///< first useful sample of each of the two primary symbols
    std::vector<std::uint32_t> halfFrames; ///< which half-frame each of those came from
};

[[nodiscard]] Frame makeFrame(std::uint32_t nId1, std::uint32_t nId2, const FrameGeometry& geometry, std::uint64_t seed) {
    gr::rng::Xoshiro256pp                                                    rng(seed);
    gr::algorithm::FFT<Complex, Complex, gr::algorithm::Direction::Backward> inverse;
    const std::array<std::complex<double>, gr::lte::kSignalLength>           pss  = gr::lte::pssSequence(nId2);
    const std::array<float, gr::lte::kSignalLength>                          sss0 = gr::lte::sssSequence(nId1, nId2, 0U);
    const std::array<float, gr::lte::kSignalLength>                          sss5 = gr::lte::sssSequence(nId1, nId2, 5U);

    const std::size_t symbols = geometry.symbolsPerSlot();
    // Paired spectrum carries the primary signal in the last symbol of slots 0 and 10 with the secondary one in
    // the symbol before it; unpaired spectrum carries the primary signal in symbol 2 of subframes 1 and 6 with
    // the secondary one in the last symbol of the subframe before.
    const bool        tdd     = geometry.duplex == DuplexMode::Tdd;
    const std::size_t pssSlot = tdd ? 2UZ : 0UZ;
    const std::size_t pssAt   = tdd ? 2UZ : symbols - 1UZ;
    const std::size_t sssSlot = tdd ? 1UZ : 0UZ;
    const std::size_t sssAt   = tdd ? symbols - 1UZ : symbols - 2UZ;

    Frame frame;
    frame.samples.reserve(gr::lte::kFrameSamples);

    const auto qpsk = [&rng] {
        const std::uint64_t bits  = rng();
        const float         scale = 1.f / std::numbers::sqrt2_v<float>;
        return Complex((bits & 1ULL) != 0ULL ? scale : -scale, (bits & 2ULL) != 0ULL ? scale : -scale);
    };

    for (std::size_t slot = 0UZ; slot < 20UZ; ++slot) {
        const std::uint32_t half = slot < 10UZ ? 0U : 1U;
        for (std::size_t symbol = 0UZ; symbol < symbols; ++symbol) {
            const bool isPss = (slot % 10UZ) == pssSlot && symbol == pssAt;
            const bool isSss = (slot % 10UZ) == sssSlot && symbol == sssAt;

            std::vector<Complex> bins(gr::lte::kSymbolSamples, Complex(0.f, 0.f));
            for (std::size_t n = 0UZ; n < 72UZ; ++n) { // subcarriers -36..-1 and +1..+36 of the central six blocks
                const std::size_t bin = n < 36UZ ? n + 92UZ : n - 35UZ;
                bins[bin]             = qpsk();
            }
            if (isPss) {
                for (std::size_t n = 0UZ; n < gr::lte::kSignalLength; ++n) {
                    bins[gr::lte::subcarrierBin(n)] = Complex(static_cast<float>(pss[n].real()), static_cast<float>(pss[n].imag()));
                }
            }
            if (isSss) {
                const std::array<float, gr::lte::kSignalLength>& values = half == 0U ? sss0 : sss5;
                for (std::size_t n = 0UZ; n < gr::lte::kSignalLength; ++n) {
                    bins[gr::lte::subcarrierBin(n)] = Complex(values[n], 0.f);
                }
            }

            std::vector<Complex> useful(gr::lte::kSymbolSamples);
            inverse.compute(bins, useful);
            // 72 unit-power subcarriers through an unnormalized inverse transform put 128*72 of energy
            // into 128 samples, so this is what makes the scene's mean per-sample power one.
            const float scale = 1.f / std::sqrt(72.f);
            for (Complex& sample : useful) {
                sample *= scale;
            }

            const std::size_t prefix = symbol == 0UZ ? geometry.firstPrefix() : geometry.otherPrefix();
            for (std::size_t n = 0UZ; n < prefix; ++n) {
                frame.samples.push_back(useful[gr::lte::kSymbolSamples - prefix + n]);
            }
            if (isPss) {
                frame.pss.push_back(frame.samples.size());
                frame.halfFrames.push_back(half);
            }
            frame.samples.insert(frame.samples.end(), useful.begin(), useful.end());
        }
    }
    return frame;
}

} // namespace

const boost::ut::suite<"LteSyncSignals"> _lteSyncSignals = [] {
    "the primary sequences are the standard's, and symmetric"_test = [] {
        const std::array<std::complex<double>, gr::lte::kSignalLength> d0  = gr::lte::pssSequence(0U);
        const auto                                                     pin = [](double u, double k) { return std::polar(1.0, -std::numbers::pi * u * k / 63.0); };

        expect(std::abs(d0[0UZ] - std::complex<double>(1.0, 0.0)) < 1e-12) << "d(0) is one";
        expect(std::abs(d0[1UZ] - pin(25.0, 2.0)) < 1e-12) << "d(1)";
        expect(std::abs(d0[30UZ] - pin(25.0, 930.0)) < 1e-12) << "d(30), the last below the puncture";
        expect(std::abs(d0[31UZ] - pin(25.0, 1056.0)) < 1e-12) << "d(31), the first above it";

        for (std::uint32_t nId2 = 0U; nId2 < 3U; ++nId2) {
            const std::array<std::complex<double>, gr::lte::kSignalLength> d = gr::lte::pssSequence(nId2);
            for (std::size_t n = 0UZ; n < gr::lte::kSignalLength; ++n) {
                expect(std::abs(d[n] - d[gr::lte::kSignalLength - 1UZ - n]) < 1e-12) << std::format("root {} is centrally symmetric at {}", nId2, n);
            }
        }

        const std::array<std::complex<double>, gr::lte::kSignalLength> d1 = gr::lte::pssSequence(1U);
        const std::array<std::complex<double>, gr::lte::kSignalLength> d2 = gr::lte::pssSequence(2U);
        for (std::size_t n = 0UZ; n < gr::lte::kSignalLength; ++n) {
            expect(std::abs(d1[n] - std::conj(d2[n])) < 1e-12) << std::format("roots 29 and 34 are conjugates at {}", n);
        }
    };

    "the primary sequences separate the three roots"_test = [] {
        const std::array<std::complex<double>, gr::lte::kSignalLength> d0 = gr::lte::pssSequence(0U);
        const std::array<std::complex<double>, gr::lte::kSignalLength> d1 = gr::lte::pssSequence(1U);
        const std::array<std::complex<double>, gr::lte::kSignalLength> d2 = gr::lte::pssSequence(2U);

        expect(approx(crossCorrelationPeak(d0, d0), 1.0, 1e-9)) << "a sequence against itself";
        expect(approx(crossCorrelationPeak(d0, d1), 0.2092, 5e-5)) << "roots 25 and 29";
        expect(approx(crossCorrelationPeak(d0, d2), 0.3844, 5e-5)) << "roots 25 and 34";
        expect(approx(crossCorrelationPeak(d1, d2), 0.2693, 5e-5)) << "roots 29 and 34";
    };

    "the shift table reproduces from its closed form"_test = [] {
        const std::array<std::pair<std::uint32_t, gr::lte::ShiftPair>, 7UZ> pins{{
            {0U, {0U, 1U}},
            {1U, {1U, 2U}},
            {29U, {29U, 30U}},
            {30U, {0U, 2U}},
            {31U, {1U, 3U}},
            {100U, {13U, 17U}},
            {167U, {2U, 9U}},
        }};
        for (const auto& [nId1, pair] : pins) {
            expect(gr::lte::m0m1(nId1) == pair) << std::format("group {}", nId1);
            expect(gr::lte::nId1FromPair(pair.m0, pair.m1) == std::optional<std::uint32_t>(nId1)) << std::format("group {} reads back", nId1);
        }

        std::set<std::pair<std::uint32_t, std::uint32_t>> seen;
        for (std::uint32_t nId1 = 0U; nId1 < gr::lte::kCellGroups; ++nId1) {
            const gr::lte::ShiftPair pair = gr::lte::m0m1(nId1);
            seen.emplace(pair.m0, pair.m1);
        }
        expect(eq(seen.size(), static_cast<std::size_t>(gr::lte::kCellGroups))) << "all 168 pairs are distinct";
    };

    "the three generators start where the standard says"_test = [] {
        const std::array<float, 10UZ> sPin{1.f, 1.f, 1.f, 1.f, -1.f, 1.f, 1.f, -1.f, 1.f, -1.f};
        const std::array<float, 10UZ> cPin{1.f, 1.f, 1.f, 1.f, -1.f, 1.f, -1.f, 1.f, -1.f, -1.f};
        const std::array<float, 10UZ> zPin{1.f, 1.f, 1.f, 1.f, -1.f, -1.f, -1.f, 1.f, 1.f, -1.f};

        const std::array<float, gr::lte::kMSequenceLength> s = gr::lte::sSequence();
        const std::array<float, gr::lte::kMSequenceLength> c = gr::lte::cSequence();
        const std::array<float, gr::lte::kMSequenceLength> z = gr::lte::zSequence();
        for (std::size_t i = 0UZ; i < 10UZ; ++i) {
            expect(eq(s[i], sPin[i])) << std::format("s at {}", i);
            expect(eq(c[i], cPin[i])) << std::format("c at {}", i);
            expect(eq(z[i], zPin[i])) << std::format("z at {}", i);
        }
    };

    "the secondary sequences separate the identities and the half-frames"_test = [] {
        const auto inner = [](const std::array<float, gr::lte::kSignalLength>& a, const std::array<float, gr::lte::kSignalLength>& b) {
            float sum = 0.f;
            for (std::size_t n = 0UZ; n < gr::lte::kSignalLength; ++n) {
                sum += a[n] * b[n];
            }
            return sum;
        };
        const std::array<float, gr::lte::kSignalLength> own = gr::lte::sssSequence(0U, 0U, 0U);
        expect(eq(inner(own, own), 62.f)) << "against itself";
        expect(eq(inner(own, gr::lte::sssSequence(1U, 0U, 0U)), 6.f)) << "against the next group";
        expect(eq(inner(own, gr::lte::sssSequence(0U, 0U, 5U)), 6.f)) << "against its own second half-frame";
    };

    "every identity and both half-frames read back from clean values"_test = [] {
        std::size_t checked = 0UZ;
        for (std::uint32_t nId1 = 0U; nId1 < gr::lte::kCellGroups; ++nId1) {
            for (std::uint32_t nId2 = 0U; nId2 < 3U; ++nId2) {
                for (const std::uint32_t subframe : {0U, 5U}) {
                    const std::array<float, gr::lte::kSignalLength> values  = gr::lte::sssSequence(nId1, nId2, subframe);
                    const gr::lte::SssReading                       reading = gr::lte::decodeSssValues(std::span<const float, gr::lte::kSignalLength>(values), nId2);
                    if (!reading.found || reading.nId1 != nId1 || reading.subframe != subframe || std::abs(reading.metric - 1.f) > 1e-5f) {
                        expect(false) << std::format("cell {} ({},{}) subframe {} read back as found={} group={} subframe={} rho={:.6f}", gr::lte::cellIdentity(nId1, nId2), nId1, nId2, subframe, reading.found, reading.nId1, reading.subframe, reading.metric);
                    }
                    ++checked;
                }
            }
        }
        expect(eq(checked, 2UZ * static_cast<std::size_t>(gr::lte::kCellIdentities))) << "504 identities in both forms";
    };

    "the geometry states the standard's offsets"_test = [] {
        expect(eq(gr::lte::kStructures[0UZ].secondaryOffset(), std::ptrdiff_t{-137})) << "paired, normal prefix";
        expect(eq(gr::lte::kStructures[1UZ].secondaryOffset(), std::ptrdiff_t{-160})) << "paired, extended prefix";
        expect(eq(gr::lte::kStructures[2UZ].secondaryOffset(), std::ptrdiff_t{-412})) << "unpaired, normal prefix";
        expect(eq(gr::lte::kStructures[3UZ].secondaryOffset(), std::ptrdiff_t{-480})) << "unpaired, extended prefix";
        expect(eq(gr::lte::kStructures[0UZ].slotOffset(), std::ptrdiff_t{-832}));
        expect(eq(gr::lte::kStructures[1UZ].slotOffset(), std::ptrdiff_t{-832})) << "both prefix types put the slot 832 back";
        expect(eq(gr::lte::kStructures[0UZ].frameStartOffset(1U), std::ptrdiff_t{-832 - 9600}));
        expect(eq(gr::lte::kStructures[2UZ].frameStartOffset(0U), std::ptrdiff_t{-284 - 1920}));
        expect(eq(gr::lte::kStructures[3UZ].frameStartOffset(1U), std::ptrdiff_t{-352 - 1920 - 9600}));
        for (const FrameGeometry& geometry : gr::lte::kStructures) {
            const std::size_t slot = geometry.symbolsPerSlot() * gr::lte::kSymbolSamples + geometry.firstPrefix() + (geometry.symbolsPerSlot() - 1UZ) * geometry.otherPrefix();
            expect(eq(slot, gr::lte::kSlotSamples)) << "a slot closes at 960 samples whichever prefix is in use";
        }
    };

    "the two kernels identify a clean frame under every structure"_test = [] {
        // Two cells that exercise both ends of the group range and every root.
        const std::array<std::pair<std::uint32_t, std::uint32_t>, 3UZ> cells{{{0U, 0U}, {100U, 1U}, {167U, 2U}}};
        gr::lte::PssCorrelator                                         correlator(0.f);
        gr::lte::SssDecoder                                            decoder;
        std::array<double, 3UZ>                                        worst{};
        std::array<double, 3UZ>                                        squared{};
        std::array<std::size_t, 3UZ>                                   drawn{};

        // Eight scenes per cell and structure. The floor the estimate carries is the scene's own random data, so a
        // single draw of it says nothing about the bound: what is asserted has to hold over many of them.
        for (const FrameGeometry& geometry : gr::lte::kStructures) {
            for (const auto& [nId1, nId2] : cells) {
                for (std::uint64_t draw = 0ULL; draw < 8ULL; ++draw) {
                    const Frame frame = makeFrame(nId1, nId2, geometry, 0x5eedULL + nId1 + 0x100ULL * draw);

                    // The window the consuming block would hand the kernels: the first half-frame, with the context
                    // before it that the furthest secondary hypothesis needs and the samples after it that a symbol
                    // beginning at the last position occupies.
                    std::vector<Complex> window(gr::lte::kMaxSecondaryLookBehind, Complex(0.f, 0.f));
                    window.insert(window.end(), frame.samples.begin(), frame.samples.begin() + static_cast<std::ptrdiff_t>(gr::lte::kHalfFrameSamples + gr::lte::kSymbolSamples));

                    const std::array<gr::lte::PssDetection, 3UZ> found = correlator.search(window, gr::lte::kHalfFrameSamples);
                    const gr::lte::PssDetection&                 best  = found[nId2];
                    expect(eq(best.position, frame.pss[0UZ] + gr::lte::kMaxSecondaryLookBehind)) << std::format("cell ({},{}) draw {} primary position", nId1, nId2, draw);
                    // The clean-scene ceiling is 62, not 128: the scene occupies 72 of the 128 bins, so the window
                    // mean of the correlation power is 128*128/72 rather than 128 and the ratio is 62*128/128 = 62.
                    expect(best.metric > 55.f) << std::format("cell ({},{}) draw {} primary metric {:.1f} on a clean frame", nId1, nId2, draw, best.metric);
                    // The two half-symbol correlations are not orthogonal to the ten subcarriers outside the
                    // sequence, which the full-symbol correlation is, so the ten of them leak a deterministic error
                    // into the estimate that no signal-to-noise ratio removes. Its bound is the root's own: 530 Hz
                    // for root 25, 228 Hz for roots 29 and 34. What is asserted is the 600 Hz the identifier's
                    // clean-frame criterion allows, and the arm reports the distribution behind it.
                    expect(std::abs(best.frequencyHz) < 600.f) << std::format("cell ({},{}) draw {} reports no offset, got {:.1f} Hz", nId1, nId2, draw, best.frequencyHz);
                    worst[nId2] = std::max(worst[nId2], static_cast<double>(std::abs(best.frequencyHz)));
                    squared[nId2] += static_cast<double>(best.frequencyHz) * static_cast<double>(best.frequencyHz);
                    ++drawn[nId2];

                    const gr::lte::SssDecision decision = decoder.decode(window, best.position, nId2, best.frequencyHz);
                    expect(decision.confirmed) << std::format("cell ({},{}) draw {} secondary confirms", nId1, nId2, draw);
                    expect(eq(decision.nId1, nId1));
                    expect(eq(decision.halfFrame, 0U));
                    expect(decision.geometry.duplex == geometry.duplex) << "the duplex mode the spacing implies";
                    expect(decision.geometry.cyclicPrefix == geometry.cyclicPrefix) << "the prefix type the spacing implies";
                    expect(decision.metric > 0.99f) << std::format("cell ({},{}) draw {} secondary metric {:.4f}", nId1, nId2, draw, decision.metric);

                    const std::ptrdiff_t frameStart = static_cast<std::ptrdiff_t>(best.position) + decision.geometry.frameStartOffset(decision.halfFrame);
                    expect(eq(frameStart, static_cast<std::ptrdiff_t>(gr::lte::kMaxSecondaryLookBehind))) << std::format("cell ({},{}) draw {} frame start", nId1, nId2, draw);
                }
            }
        }
        for (std::uint32_t root = 0U; root < 3U; ++root) {
            std::println("clean-frame carrier offset, root {}: {} draws, {:.1f} Hz rms, {:.1f} Hz at the extreme", gr::lte::kPssRoots[root], drawn[root], drawn[root] == 0UZ ? 0. : std::sqrt(squared[root] / static_cast<double>(drawn[root])), worst[root]);
        }
    };

    "the second half-frame reads as the second half-frame"_test = [] {
        gr::lte::PssCorrelator correlator(0.f);
        gr::lte::SssDecoder    decoder;
        for (const FrameGeometry& geometry : gr::lte::kStructures) {
            const Frame frame = makeFrame(42U, 1U, geometry, 0xabcdULL);

            std::vector<Complex>                         window(frame.samples.begin() + static_cast<std::ptrdiff_t>(gr::lte::kHalfFrameSamples - gr::lte::kMaxSecondaryLookBehind), frame.samples.end());
            const std::array<gr::lte::PssDetection, 3UZ> found = correlator.search(window, gr::lte::kHalfFrameSamples);
            const gr::lte::PssDetection&                 best  = found[1UZ];
            expect(eq(best.position + gr::lte::kHalfFrameSamples - gr::lte::kMaxSecondaryLookBehind, frame.pss[1UZ]));

            const gr::lte::SssDecision decision = decoder.decode(window, best.position, 1U, best.frequencyHz);
            expect(decision.confirmed);
            expect(eq(decision.nId1, 42U));
            expect(eq(decision.halfFrame, 1U)) << "the subframe-5 form is what says so";
        }
    };

    "the fractional estimator reads a carrier offset back"_test = [] {
        const FrameGeometry    geometry{DuplexMode::Fdd, CyclicPrefix::Normal};
        gr::lte::PssCorrelator correlator(0.f);
        double                 worst = 0.;

        // Eight scenes per offset: the error the ten subcarriers outside the sequence leak into the two
        // half-symbol correlations is the scene's own, so an assertion over one draw of it holds by its seed.
        for (std::uint64_t draw = 0ULL; draw < 8ULL; ++draw) {
            const Frame frame = makeFrame(7U, 2U, geometry, 0x1234ULL + 0x100ULL * draw);
            for (const float offset : {-3000.f, -1000.f, 0.f, 1000.f, 3000.f}) {
                std::vector<Complex> window(gr::lte::kMaxSecondaryLookBehind, Complex(0.f, 0.f));
                window.insert(window.end(), frame.samples.begin(), frame.samples.begin() + static_cast<std::ptrdiff_t>(gr::lte::kHalfFrameSamples + gr::lte::kSymbolSamples));
                for (std::size_t n = 0UZ; n < window.size(); ++n) {
                    const double turn = 2. * std::numbers::pi * static_cast<double>(offset) * static_cast<double>(n) / static_cast<double>(gr::lte::kSampleRate);
                    window[n] *= Complex(static_cast<float>(std::cos(turn)), static_cast<float>(std::sin(turn)));
                }
                const std::array<gr::lte::PssDetection, 3UZ> found = correlator.search(window, gr::lte::kHalfFrameSamples);
                expect(eq(found[2UZ].position, frame.pss[0UZ] + gr::lte::kMaxSecondaryLookBehind)) << std::format("offset {} Hz draw {}", offset, draw);
                const double error = static_cast<double>(found[2UZ].frequencyHz) - static_cast<double>(offset);
                worst              = std::max(worst, std::abs(error));
                expect(std::abs(error) < 600.) << std::format("offset {} Hz draw {} read back as {:.1f}", offset, draw, found[2UZ].frequencyHz);
            }
        }
        std::println("the fractional estimator over 40 clean scenes: {:.1f} Hz at the extreme", worst);
    };

    "the hypothesis grid is half a subcarrier and covers the stated width"_test = [] {
        const gr::lte::PssCorrelator none(0.f);
        expect(eq(none.hypotheses(), 1UZ));
        expect(eq(none.hypothesisFrequency(0UZ), 0.f));

        const gr::lte::PssCorrelator wide(45'000.f);
        expect(eq(wide.hypotheses(), 13UZ)) << "2*ceil(45000/7500)+1";
        expect(eq(wide.hypothesisFrequency(0UZ), -45'000.f));
        expect(eq(wide.hypothesisFrequency(12UZ), 45'000.f));
        expect(eq(gr::lte::PssCorrelator(1.f).hypotheses(), 3UZ)) << "any search at all is at least one step either way";
    };
};

int main() { /* not needed for UT */ }
