#include <boost/ut.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <format>
#include <numbers>
#include <span>
#include <vector>

#include <gnuradio-4.0/algorithm/fourier/PhaseVocoder.hpp>
#include <gnuradio-4.0/algorithm/fourier/fft.hpp>

namespace {

using CF = std::complex<float>;

/// A complex exponential at `cycles` cycles per sample, unit amplitude.
[[nodiscard]] std::vector<CF> tone(std::size_t count, double cycles, double amplitude = 1.0) {
    std::vector<CF> out(count);
    for (std::size_t n = 0UZ; n < count; ++n) {
        const double phase = 2.0 * std::numbers::pi * cycles * static_cast<double>(n);
        out[n]             = static_cast<float>(amplitude) * CF{static_cast<float>(std::cos(phase)), static_cast<float>(std::sin(phase))};
    }
    return out;
}

/// The frequency (cycles per sample, in [0, 1)) of the strongest bin of a windowed transform.
[[nodiscard]] double strongestBin(std::span<const CF> x) {
    std::size_t size = 1UZ;
    while (size * 2UZ <= x.size()) {
        size *= 2UZ;
    }
    gr::algorithm::FFT<CF> fft;
    std::vector<CF>        windowed(size);
    for (std::size_t i = 0UZ; i < size; ++i) {
        const double w = 0.5 - 0.5 * std::cos(2.0 * std::numbers::pi * static_cast<double>(i) / static_cast<double>(size));
        windowed[i]    = x[x.size() - size + i] * static_cast<float>(w);
    }
    std::vector<CF> spectrum(size);
    fft.compute(windowed, spectrum);
    std::size_t best = 0UZ;
    for (std::size_t k = 1UZ; k < size; ++k) {
        if (std::abs(spectrum[k]) > std::abs(spectrum[best])) {
            best = k;
        }
    }
    return static_cast<double>(best) / static_cast<double>(size);
}

/// A complex tone's amplitude IS its RMS, so level survives measurement without window calibration.
[[nodiscard]] double rmsDb(std::span<const CF> x) {
    double power = 0.0;
    for (const CF v : x) {
        power += static_cast<double>(std::norm(v));
    }
    return 10.0 * std::log10(power / static_cast<double>(x.size()));
}

[[nodiscard]] std::vector<CF> run(gr::algorithm::PhaseVocoder& engine, std::span<const CF> input, std::size_t chunk = 0UZ) {
    std::vector<CF>   out;
    const std::size_t stride = chunk == 0UZ ? input.size() : chunk;
    for (std::size_t base = 0UZ; base < input.size(); base += stride) {
        engine.process(input.subspan(base, std::min(stride, input.size() - base)), out);
    }
    return out;
}

} // namespace

const boost::ut::suite<"PhaseVocoder"> phaseVocoderTests = [] {
    using namespace boost::ut;
    using gr::algorithm::PhaseVocoder;

    constexpr std::size_t kFrame = 1024UZ;
    constexpr std::size_t kHopIn = 256UZ;

    "a pure tone lands at its frequency divided, level held"_test = [] {
        for (const std::size_t divisor : {2UZ, 8UZ, 32UZ}) {
            PhaseVocoder engine;
            engine.configure(kFrame, kHopIn, kHopIn / divisor);
            const double cycles = 0.21; // well inside the band at every divisor
            const auto   out    = run(engine, tone(64UZ * kFrame, cycles));

            // The output stream is declared at 1/divisor of the input rate, so the divided
            // frequency in output samples is the input frequency unchanged: transposition
            // shows up as the same normalized frequency over 1/divisor as many samples.
            const std::span<const CF> tail{std::span<const CF>(out).subspan(out.size() / 2UZ)};
            const double              peak = strongestBin(tail);
            const auto                bins = static_cast<double>(kFrame);
            expect(lt(std::abs(peak - cycles), 1.0 / bins)) << std::format("divisor {}: peak at {} for {}", divisor, peak, cycles);
            expect(lt(std::abs(rmsDb(tail)), 0.5)) << std::format("divisor {}: level {} dB against the unit tone", divisor, rmsDb(tail));
        }
    };

    "transposition, not translation: a harmonic pair stays harmonic"_test = [] {
        PhaseVocoder engine;
        engine.configure(kFrame, kHopIn, kHopIn / 8UZ);
        const double f = 0.03;
        auto         x = tone(64UZ * kFrame, f, 0.7);
        const auto   h = tone(64UZ * kFrame, 3.0 * f, 0.7);
        for (std::size_t i = 0UZ; i < x.size(); ++i) {
            x[i] += h[i];
        }
        const auto out = run(engine, x);

        // Each tone keeps its own normalized frequency, so the pair's ratio survives exactly
        // where an offset would have destroyed it.
        gr::algorithm::FFT<CF> fft;
        std::vector<CF>        tail(out.end() - static_cast<std::ptrdiff_t>(kFrame), out.end());
        std::vector<CF>        spectrum(kFrame);
        fft.compute(tail, spectrum);
        const auto   bin   = [&](double cycles) { return static_cast<double>(std::abs(spectrum[static_cast<std::size_t>(std::lround(cycles * static_cast<double>(kFrame)))])); };
        const double floor = static_cast<double>(std::abs(spectrum[kFrame / 2UZ]));
        expect(gt(bin(f), 100.0 * floor)) << "the fundamental stands";
        expect(gt(bin(3.0 * f), 100.0 * floor)) << "and its third harmonic beside it";
    };

    "the steady-state output count is the input count divided"_test = [] {
        for (const std::size_t divisor : {4UZ, 16UZ}) {
            PhaseVocoder engine;
            engine.configure(kFrame, kHopIn, kHopIn / divisor);
            const std::size_t n    = 200UZ * kHopIn;
            const auto        out  = run(engine, tone(n, 0.1));
            const auto        want = static_cast<double>(n) / static_cast<double>(divisor);
            expect(lt(std::abs(static_cast<double>(out.size()) - want), static_cast<double>(kFrame))) << std::format("divisor {}: {} of {}", divisor, out.size(), want);
        }
    };

    "the chunking is not observable"_test = [] {
        const auto   x = tone(16UZ * kFrame, 0.13);
        PhaseVocoder reference;
        reference.configure(kFrame, kHopIn, kHopIn / 4UZ);
        const auto want = run(reference, x);
        for (const std::size_t chunk : {1UZ, 7UZ, 997UZ}) {
            PhaseVocoder engine;
            engine.configure(kFrame, kHopIn, kHopIn / 4UZ);
            expect(that % (run(engine, x, chunk) == want)) << std::format("chunk {}", chunk);
        }
    };

    "reset returns the engine to the primed-from-nothing state"_test = [] {
        const auto   x = tone(8UZ * kFrame, 0.07);
        PhaseVocoder engine;
        engine.configure(kFrame, kHopIn, kHopIn / 4UZ);
        const auto first = run(engine, x);
        engine.reset();
        expect(that % (run(engine, x) == first)) << "the same stream replays to the same output";
    };

    "equal hops replay the stream exactly"_test = [] {
        PhaseVocoder engine;
        engine.configure(kFrame, kHopIn, kHopIn);
        const auto x   = tone(3UZ * kFrame + 17UZ, 0.05);
        const auto out = run(engine, x);
        expect(that % (out == x)) << "identity is a copy, with no latency";
    };

    "refusals fire by name"_test = [] {
        PhaseVocoder engine;
        expect(throws([&] { engine.configure(1000UZ, 250UZ, 25UZ); })) << "frame must be a power of two";
        expect(throws([&] { engine.configure(1024UZ, 0UZ, 1UZ); })) << "hops must be positive";
        expect(throws([&] { engine.configure(1024UZ, 2048UZ, 256UZ); })) << "a hop past the frame leaves gaps";
    };
};

int main() { /* not needed for UT */ }
