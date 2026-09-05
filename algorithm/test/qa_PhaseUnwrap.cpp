#include <boost/ut.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <numbers>
#include <print>
#include <span>
#include <stdexcept>
#include <vector>

#include <gnuradio-4.0/algorithm/measurement/PhaseUnwrap.hpp>

using namespace boost::ut;
using gr::measurement::CycleUnwrapper;
using gr::measurement::UnwrapOrigin;
using gr::measurement::unwrapOriginFrom;
using gr::measurement::unwrapOriginName;

namespace {

constexpr double kTwoPi = 2. * std::numbers::pi_v<double>;
using Complex           = std::complex<float>;

[[nodiscard]] bool longRun() { return std::getenv("ENABLE_LONG_TESTS") != nullptr; }

/// A tone at @p f0 cycles per sample. The phase of sample `k` is computed from `k` itself rather than
/// accumulated, so the stream the unwrapper is judged on carries no accumulation error of its own.
[[nodiscard]] std::vector<Complex> tone(double f0, std::size_t n, std::size_t first = 0UZ) {
    std::vector<Complex> out(n);
    for (std::size_t k = 0UZ; k < n; ++k) {
        const double phase = kTwoPi * f0 * static_cast<double>(first + k);
        out[k]             = Complex{static_cast<float>(std::cos(phase)), static_cast<float>(std::sin(phase))};
    }
    return out;
}

/// What a `float` accumulator of total radians would read on the same stream, and what a `double` one would:
/// the two designs this one exists instead of, run side by side so the reason is demonstrated and not asserted.
struct FloatComparison {
    float  asFloat{0.f};
    double asDouble{0.};
};

void accumulate(FloatComparison& into, float step) noexcept {
    into.asFloat += step;
    into.asDouble += static_cast<double>(step);
}

} // namespace

const boost::ut::suite<"PhaseUnwrap"> _phase_unwrap = [] {
    "the two spellings, and nothing else"_test = [] {
        expect(unwrapOriginFrom("first_sample").value() == UnwrapOrigin::first_sample);
        expect(unwrapOriginFrom("zero").value() == UnwrapOrigin::zero);
        expect(!unwrapOriginFrom("origin").has_value());
        expect(unwrapOriginName(UnwrapOrigin::zero) == "zero");
        expect(throws<std::invalid_argument>([] { (void)CycleUnwrapper(UnwrapOrigin::zero, 0.); })) << "a max step fraction of zero is not in (0, 1]";
        expect(throws<std::invalid_argument>([] { (void)CycleUnwrapper(UnwrapOrigin::zero, 1.5); }));
        expect(nothrow([] { (void)CycleUnwrapper(UnwrapOrigin::zero, 1.); })) << "and the bound itself is admitted";
    };

    "the count is exact where a float accumulator has stopped working"_test = [] {
        // 0.4 cycles per sample is 2 turns every 5 samples, so `5*C/2 + 1` samples land exactly on turn `C`
        // with no sample at +/-pi to round either way.
        const std::int64_t    turns   = longRun() ? 1'000'000'000LL : (1LL << 21);
        const std::size_t     samples = static_cast<std::size_t>(5LL * turns / 2LL) + 1UZ;
        constexpr std::size_t kChunk  = 1UZ << 16;

        CycleUnwrapper  unwrapper;
        FloatComparison naive;

        std::vector<std::int64_t> cycles(kChunk);
        std::vector<float>        phase(kChunk);
        float                     previous = 0.f;
        for (std::size_t at = 0UZ; at < samples; at += kChunk) {
            const std::size_t          n  = std::min(kChunk, samples - at);
            const std::vector<Complex> in = tone(0.4, n, at);
            unwrapper.process(in, std::span<std::int64_t>(cycles).first(n), std::span<float>(phase).first(n));
            for (std::size_t k = 0UZ; k < n; ++k) {
                if (at + k > 0UZ) {
                    const float delta = phase[k] - previous;
                    accumulate(naive, (delta > CycleUnwrapper::kPi) ? (delta - 2.f * CycleUnwrapper::kPi) : ((delta < -CycleUnwrapper::kPi) ? (delta + 2.f * CycleUnwrapper::kPi) : delta));
                }
                previous = phase[k];
            }
        }

        const double naiveCycles  = static_cast<double>(naive.asFloat) / kTwoPi;
        const double doubleCycles = naive.asDouble / kTwoPi;
        std::println("after {} samples: cycles() = {}, a float accumulator reads {:.3f} turns (off by {:.3f}), a double one {:.6f} (off by {:.3e}){}", samples, unwrapper.cycles(), naiveCycles, naiveCycles - static_cast<double>(turns), doubleCycles, doubleCycles - static_cast<double>(turns), longRun() ? " [ENABLE_LONG_TESTS]" : "");

        expect(unwrapper.cycles() == turns) << "the integer count is exact at" << turns << "turns";
        expect(std::abs(static_cast<double>(unwrapper.phase())) < 4. * 2.38e-7) << "and the residual is zero to a float's spacing at pi";
        expect(std::abs(naiveCycles - static_cast<double>(turns)) > 1.) << "while a float accumulator over the same stream is wrong by more than a whole cycle";
        expect(unwrapper.nSuspectSteps() == 0ULL) << "0.4 cycles per sample steps 0.8*pi, inside the default fraction";
        expect(unwrapper.nSaturations() == 0ULL);
    };

    "past Nyquist the unwrap is the alias, which is the only thing it can be"_test = [] {
        constexpr std::int64_t kTurns  = 1LL << 14;
        const std::size_t      samples = static_cast<std::size_t>(5LL * kTurns / 2LL) + 1UZ;

        CycleUnwrapper            forward;
        std::vector<std::int64_t> cycles(samples);
        std::vector<float>        phase(samples);
        forward.process(tone(0.4, samples), cycles, phase);
        expect(forward.cycles() == kTurns) << "0.4 cycles per sample unwraps exactly";
        expect(forward.nSuspectSteps() == 0ULL);

        // 0.6 cycles per sample advances 1.2*pi per sample, which the wrapped difference reports as -0.8*pi.
        // Two turn counts produce the same samples and nothing operating on the samples alone can choose
        // between them: this is Nyquist restated for phase, and the aliased answer is the correct output of a
        // correct algorithm on an input that does not determine the question.
        CycleUnwrapper aliased;
        aliased.process(tone(0.6, samples), cycles, phase);
        expect(aliased.cycles() == -kTurns) << "0.6 cycles per sample reads the alias at -0.4, exactly";
        expect(aliased.nSuspectSteps() == 0ULL) << "and the step is 0.8*pi either way, so the observability hook does not fire -- it is a hook, not a detector";

        // What does fire is a step that comes close to pi. At 0.47 cycles per sample the step is 0.94*pi, past
        // the default 0.9, while the unwrap is still exactly right.
        CycleUnwrapper close;
        close.process(tone(0.47, 1'000UZ), cycles, phase);
        expect(close.nSuspectSteps() == 999ULL) << "every step but the first is flagged";
        expect(close.lastSuspectIndex().value() == 999ULL) << "and the last one names its own absolute index";

        CycleUnwrapper quiet;
        expect(!quiet.lastSuspectIndex().has_value()) << "a stream with no suspect step names none";
    };

    "the two origins differ only in where they start"_test = [] {
        constexpr std::size_t kSamples = 4'096UZ;
        const auto            in       = tone(0.13, kSamples, 3UZ); // a first sample whose phase is not zero

        std::vector<std::int64_t> cyclesA(kSamples);
        std::vector<float>        phaseA(kSamples);
        CycleUnwrapper            asIs(UnwrapOrigin::first_sample);
        asIs.process(in, cyclesA, phaseA);

        std::vector<std::int64_t> cyclesB(kSamples);
        std::vector<float>        phaseB(kSamples);
        CycleUnwrapper            zeroed(UnwrapOrigin::zero);
        zeroed.process(in, cyclesB, phaseB);

        expect(phaseA[0] == std::atan2(in[0].imag(), in[0].real())) << "first_sample reports the first phase as it is";
        expect(phaseB[0] == 0.f) << "zero starts at exactly zero";
        expect(cyclesA[0] == 0LL && cyclesB[0] == 0LL);

        // Neither changes the differences, which is what a consumer usually wants.
        double worst = 0.;
        for (std::size_t k = 1UZ; k < kSamples; ++k) {
            const double a = kTwoPi * static_cast<double>(cyclesA[k] - cyclesA[k - 1UZ]) + static_cast<double>(phaseA[k] - phaseA[k - 1UZ]);
            const double b = kTwoPi * static_cast<double>(cyclesB[k] - cyclesB[k - 1UZ]) + static_cast<double>(phaseB[k] - phaseB[k - 1UZ]);
            worst          = std::max(worst, std::abs(a - b));
        }
        expect(lt(worst, 1e-6)) << "the two origins give the same phase differences, worst" << worst;
    };

    "a gap resets the count, and says it did"_test = [] {
        constexpr std::size_t kSamples = 2'000UZ;

        CycleUnwrapper            unwrapper;
        std::vector<std::int64_t> cycles(kSamples);
        std::vector<float>        phase(kSamples);

        unwrapper.process(tone(0.1, kSamples), cycles, phase);
        const std::int64_t before = unwrapper.cycles();
        expect(before > 100LL) << "the count has run up";
        expect(unwrapper.nResets() == 0ULL);

        unwrapper.markDiscontinuity();
        expect(unwrapper.cycles() == 0LL) << "after a gap the count is no longer the number of turns the signal made, so it starts again";
        expect(unwrapper.nResets() == 1ULL) << "and the discontinuity is counted rather than hidden";
        expect(!unwrapper.started());

        unwrapper.process(tone(0.1, kSamples, 5'000UZ), cycles, phase);
        expect(unwrapper.cycles() == before) << "and the second stretch counts its own turns";
        expect(unwrapper.nResets() == 1ULL);

        // A caller who would rather keep the count through a gap simply does not call it, and gets exactly that.
        CycleUnwrapper carried;
        carried.process(tone(0.1, kSamples), cycles, phase);
        carried.process(tone(0.1, kSamples, kSamples), cycles, phase);
        expect(carried.cycles() == 2LL * before) << "an uninterrupted count keeps running";
        expect(carried.nResets() == 0ULL);
    };

    "the stream does not depend on how it is chunked"_test = [] {
        constexpr std::size_t kSamples = 12'000UZ;
        const auto            in       = tone(0.37, kSamples);

        CycleUnwrapper            whole;
        std::vector<std::int64_t> cyclesRef(kSamples);
        std::vector<float>        phaseRef(kSamples);
        whole.process(in, cyclesRef, phaseRef);

        for (const std::size_t chunk : {1UZ, 7UZ, 1'000UZ, 12'345UZ}) {
            CycleUnwrapper            pieced;
            std::vector<std::int64_t> cycles(kSamples);
            std::vector<float>        phase(kSamples);
            for (std::size_t at = 0UZ; at < kSamples; at += chunk) {
                const std::size_t n = std::min(chunk, kSamples - at);
                pieced.process(std::span<const Complex>(in).subspan(at, n), std::span<std::int64_t>(cycles).subspan(at, n), std::span<float>(phase).subspan(at, n));
            }
            expect(that % std::ranges::equal(cycles, cyclesRef)) << std::format("chunk {} changed the count", chunk);
            expect(that % std::ranges::equal(phase, phaseRef)) << std::format("chunk {} changed the phase", chunk);
            expect(pieced.nSuspectSteps() == whole.nSuspectSteps());
        }
    };

    "the parts are handed over separately, and the convenience says its own spacing"_test = [] {
        constexpr std::size_t kSamples = 101UZ;

        CycleUnwrapper            unwrapper;
        std::vector<std::int64_t> cycles(kSamples);
        std::vector<float>        phase(kSamples);
        unwrapper.process(tone(0.1, kSamples), cycles, phase);

        expect(unwrapper.cycles() == 10LL) << "0.1 cycles per sample over 100 steps is ten turns";
        expect(unwrapper.processed() == kSamples);
        expect(approx(unwrapper.unwrappedRadians(), kTwoPi * 10., 1e-5)) << "and the convenience reader sums the two parts for the current sample";

        expect(throws<std::invalid_argument>([&] {
            std::vector<std::int64_t> tooFew(3UZ);
            unwrapper.process(tone(0.1, 4UZ), tooFew, phase);
        })) << "one output short of the samples";

        unwrapper.reset();
        expect(unwrapper.cycles() == 0LL);
        expect(unwrapper.processed() == 0ULL);
        expect(unwrapper.nResets() == 0ULL) << "reset clears the counters where markDiscontinuity keeps them";
        expect(!unwrapper.started());

        // Saturation is a claim about arithmetic and not about a run: at the fastest a stream can turn and
        // still be unwrappable, `2^63 - 1` cycles takes 584.55 years at 10^9 S/s, so no test reaches it. What
        // is asserted is that nothing on the way there trips the counter.
        expect(unwrapper.nSaturations() == 0ULL);
    };
};

int main() { /* not needed for UT */ }
