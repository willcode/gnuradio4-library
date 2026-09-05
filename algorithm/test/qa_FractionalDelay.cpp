#include <boost/ut.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <numbers>
#include <print>
#include <span>
#include <stdexcept>
#include <vector>

#include <gnuradio-4.0/algorithm/filter/FractionalDelay.hpp>
#include <gnuradio-4.0/algorithm/timing/FrequencySchedule.hpp>

using namespace boost::ut;
using gr::filter::arbitraryInterpolationError;
using gr::filter::designFractionalDelay;
using gr::filter::FractionalDelayLine;
using gr::filter::fractionalDelayQ32;
using gr::filter::kArbitraryOne;
using gr::filter::kMaxFractionalDelaySamples;
using gr::timing::DelaySchedule;
using gr::timing::kSpeedOfLight;
using gr::timing::offsetFor;
using gr::timing::SampleClock;

namespace {

constexpr double kTwoPi   = 2. * std::numbers::pi_v<double>;
constexpr double kRolloff = 0.2;

using Complex = std::complex<float>;

/// A unit tone at @p f0 cycles per sample, which is what a fractional delay's contract is written against: a
/// delay of `d` samples is a phase rotation of `-2*pi*f0*d` and nothing else.
[[nodiscard]] std::vector<Complex> tone(double f0, std::size_t n) {
    std::vector<Complex> out(n);
    for (std::size_t k = 0UZ; k < n; ++k) {
        const double phase = kTwoPi * f0 * static_cast<double>(k);
        out[k]             = Complex{static_cast<float>(std::cos(phase)), static_cast<float>(std::sin(phase))};
    }
    return out;
}

/// The prototype's delivered passband ripple as a linear amplitude.
[[nodiscard]] double rippleFloor(double rippleDb) { return std::pow(10., rippleDb / 20.) - 1.; }

/// The prototype's delivered stopband as a linear amplitude: the tone's own images land there, so it is the
/// third term of the envelope beside the interpolation bound and the ripple, and not a fourth thing.
[[nodiscard]] double stopbandFloor(double stopbandDb) { return std::pow(10., stopbandDb / 20.); }

/// The slope of a linear fit through @p y against its own index, least squares, closed form.
[[nodiscard]] double slopeOf(std::span<const double> y) {
    const double n     = static_cast<double>(y.size());
    const double meanX = 0.5 * (n - 1.);
    double       meanY = 0.;
    for (const double value : y) {
        meanY += value;
    }
    meanY /= n;

    double covariance = 0.;
    double variance   = 0.;
    for (std::size_t k = 0UZ; k < y.size(); ++k) {
        const double dx = static_cast<double>(k) - meanX;
        covariance += dx * (y[k] - meanY);
        variance += dx * dx;
    }
    return covariance / variance;
}

} // namespace

const boost::ut::suite<"FractionalDelay"> _fractional_delay = [] {
    "the error bound is the resampler's, at the sizes the table states"_test = [] {
        // The five rows of the closed form, so a change to it is caught here rather than in a delay measurement
        // whose envelope it silently widened.
        struct Row {
            std::size_t bank;
            double      order0;
            double      order1;
            double      order3;
        };
        constexpr Row rows[] = {{8UZ, 1.571e-1, 1.234e-2, 2.283e-4}, {16UZ, 7.854e-2, 3.084e-3, 1.427e-5}, {32UZ, 3.927e-2, 7.711e-4, 8.918e-7}, {64UZ, 1.964e-2, 1.928e-4, 5.574e-8}, {128UZ, 9.818e-3, 4.819e-5, 3.484e-9}};

        for (const Row& row : rows) {
            expect(approx(arbitraryInterpolationError(row.bank, kRolloff, 0), row.order0, 1e-3 * row.order0)) << "L =" << row.bank << "at order 0";
            expect(approx(arbitraryInterpolationError(row.bank, kRolloff, 1), row.order1, 1e-3 * row.order1)) << "L =" << row.bank << "at order 1";
            expect(approx(arbitraryInterpolationError(row.bank, kRolloff, 3), row.order3, 1e-3 * row.order3)) << "L =" << row.bank << "at order 3";
        }
    };

    "a constant delay is a phase rotation, inside the bound the bank was cut for"_test = [] {
        constexpr std::size_t kSamples = 4'096UZ;

        struct Arm {
            std::size_t bank;
            int         order;
        };
        constexpr Arm arms[] = {{8UZ, 1}, {32UZ, 1}, {32UZ, 3}, {128UZ, 1}};

        for (const Arm& arm : arms) {
            const gr::filter::ResamplerDesign design = designFractionalDelay(arm.bank, kRolloff);
            const double                      interp = arbitraryInterpolationError(arm.bank, kRolloff, arm.order);
            const double                      bound  = interp + rippleFloor(design.rippleDb) + stopbandFloor(design.stopbandDb);

            FractionalDelayLine<Complex> line(arm.bank, arm.order, design.taps, 16ULL);
            const double                 latency = line.latencySamples();

            double worst = 0.;
            for (const double f0 : {0.05, 0.15, 0.25, 0.35}) {
                for (const double delay : {0., 0.25, 0.5, 0.75, 1.5, 7.125}) {
                    const std::vector<Complex> in = tone(f0, kSamples);
                    std::vector<std::uint64_t> delays(kSamples, fractionalDelayQ32(delay));
                    std::vector<Complex>       out(kSamples);

                    line.reset();
                    line.process(in, delays, out);

                    // The first `history + tapsPerArm` outputs are the filter's transient over the zeroed
                    // history, the same convention every filtering block in this tree states.
                    const std::size_t skip = line.historySamples() + line.tapsPerArm();
                    for (std::size_t k = skip; k < kSamples; ++k) {
                        const double  phase = kTwoPi * f0 * (static_cast<double>(k) - latency - delay);
                        const Complex ideal{static_cast<float>(std::cos(phase)), static_cast<float>(std::sin(phase))};
                        worst = std::max(worst, static_cast<double>(std::abs(out[k] - ideal)));
                    }
                }
            }
            std::println("L = {:3}, q = {}: worst deviation {:.6e} against the envelope {:.6e} = interpolation {:.6e} + ripple {:.6e} + stopband {:.6e}, over {} taps", arm.bank, arm.order, worst, bound, interp, rippleFloor(design.rippleDb), stopbandFloor(design.stopbandDb), design.taps.size());
            expect(lt(worst, bound)) << std::format("L = {}, q = {} misses its own bound", arm.bank, arm.order);
        }
    };

    "the consistency identity: a range rate is a frequency shift, at the ratio of the two carriers"_test = [] {
        // spec-doppler-trajectory-io criterion 6, at the kernel. A range rate `v` delays the envelope by
        // `tau(t) = tau0 + v*t/c`, so a baseband tone at `f_b` emerges at `f_b*(1 - v/c)` -- the derivation is
        // exact, not first order -- while the same `v` moves a carrier at `f_c` by `offsetFor(v, f_c)`. The two
        // shifts are therefore in the ratio `f_b/f_c`, which is the assertion that fails if either sign is wrong
        // and says which.
        constexpr double      kSampleRate = 48'000.;
        constexpr double      kToneHz     = 12'000.;
        constexpr double      kCarrierHz  = 437e6;
        constexpr double      kRangeRate  = -7'000.;
        constexpr std::size_t kSamples    = 480'000UZ; // 10 s
        constexpr double      kDelay0     = 20.;

        const double f0     = kToneHz / kSampleRate;
        const double perTap = kRangeRate / kSpeedOfLight; // dtau/dt, and therefore delay samples per sample

        const gr::filter::ResamplerDesign design = designFractionalDelay(32UZ, kRolloff);
        FractionalDelayLine<Complex>      line(32UZ, 1, design.taps, 21ULL);

        const std::vector<Complex> in = tone(f0, kSamples);
        std::vector<std::uint64_t> delays(kSamples);
        for (std::size_t k = 0UZ; k < kSamples; ++k) {
            delays[k] = fractionalDelayQ32(kDelay0 + perTap * static_cast<double>(k));
        }
        std::vector<Complex> out(kSamples);
        line.process(in, delays, out);

        const std::size_t   skip = line.historySamples() + line.tapsPerArm();
        std::vector<double> residual(kSamples - skip);
        double              turns = 0.;
        double              last  = 0.;
        for (std::size_t k = skip; k < kSamples; ++k) {
            const double  phase = kTwoPi * f0 * static_cast<double>(k);
            const Complex reference{static_cast<float>(std::cos(phase)), static_cast<float>(std::sin(phase))};
            const double  wrapped = std::arg(out[k] * std::conj(reference));
            if (k > skip) {
                const double step = wrapped - last;
                turns += (step > std::numbers::pi) ? -kTwoPi : ((step < -std::numbers::pi) ? kTwoPi : 0.);
            }
            last               = wrapped;
            residual[k - skip] = wrapped + turns;
        }

        // The residual phase is `-2*pi*f0*(latency + D0 + (v/c)*k)`, so its slope names the shift directly.
        const double measuredHz = slopeOf(residual) / kTwoPi * kSampleRate;
        const double expectedHz = -kToneHz * kRangeRate / kSpeedOfLight;
        const double carrierHz  = offsetFor(kRangeRate, kCarrierHz);

        std::println("tone shift {:.7f} Hz against the closed form {:.7f}; carrier shift {:.4f} Hz; ratio {:.6e} against f_b/f_c = {:.6e}", measuredHz, expectedHz, carrierHz, measuredHz / carrierHz, kToneHz / kCarrierHz);
        expect(approx(measuredHz, expectedHz, 0.01 * expectedHz)) << "the envelope's own shift is f_b*(1 - v/c)";
        expect(approx(measuredHz / carrierHz, kToneHz / kCarrierHz, 0.01 * kToneHz / kCarrierHz)) << "and it stands to the carrier's in the ratio of the two frequencies";
        expect(gt(carrierHz, 0.)) << "a closing pass reads high";
    };

    "a delay schedule drives the line through one span call"_test = [] {
        // The path a block takes: the schedule's own cursor walk writes seconds, one conversion makes them Q32
        // samples, and the line never sees a `double` delay at all.
        constexpr double      kSampleRate = 48'000.;
        constexpr std::size_t kSamples    = 8'192UZ;

        const std::vector<std::int64_t> times{0LL, 200'000'000LL};
        const std::vector<double>       seconds{8.339e-3, 6.0e-3};
        const DelaySchedule             schedule(times, seconds);
        const SampleClock               clock(0ULL, 0LL, 48'000ULL, 1ULL);

        std::vector<double> walked(kSamples);
        schedule.valuesFor(clock, 0ULL, walked);

        std::vector<std::uint64_t> delays(kSamples);
        fractionalDelayQ32(walked, kSampleRate, delays);
        expect(delays.front() == fractionalDelayQ32(8.339e-3 * kSampleRate)) << "the first sample carries the first knot";
        expect(delays.front() > delays.back()) << "and the table's fall shows up as a fall in the delay";

        const std::uint64_t maxWhole = delays.front() >> 32;
        expect(maxWhole == 400ULL) << "8.339 ms at 48 kS/s is 400 whole samples, got" << maxWhole;

        const gr::filter::ResamplerDesign design = designFractionalDelay(32UZ, kRolloff);
        FractionalDelayLine<Complex>      line(32UZ, 1, design.taps, maxWhole);
        expect(line.historySamples() == maxWhole + line.tapsPerArm() + 1UZ);

        const std::vector<Complex> in = tone(0.1, kSamples);
        std::vector<Complex>       out(kSamples);
        expect(nothrow([&] { line.process(in, delays, out); }));
    };

    "the stream does not depend on how it is chunked"_test = [] {
        constexpr std::size_t kSamples = 20'000UZ;

        const gr::filter::ResamplerDesign design = designFractionalDelay(32UZ, kRolloff);

        const std::vector<Complex> in = tone(0.17, kSamples);
        std::vector<std::uint64_t> delays(kSamples);
        for (std::size_t k = 0UZ; k < kSamples; ++k) {
            // A delay that moves within a chunk and across chunk boundaries, so that a per-call reset of the
            // read position would show up rather than canceling.
            delays[k] = fractionalDelayQ32(30. + 12. * std::sin(kTwoPi * static_cast<double>(k) / 7'000.));
        }

        for (const int order : {0, 1, 3}) {
            FractionalDelayLine<Complex> whole(32UZ, order, design.taps, 64ULL);
            std::vector<Complex>         reference(kSamples);
            whole.process(in, delays, reference);

            for (const std::size_t chunk : {1UZ, 7UZ, 1'000UZ, 12'345UZ}) {
                FractionalDelayLine<Complex> pieced(32UZ, order, design.taps, 64ULL);
                std::vector<Complex>         got(kSamples);
                for (std::size_t at = 0UZ; at < kSamples; at += chunk) {
                    const std::size_t n = std::min(chunk, kSamples - at);
                    pieced.process(std::span<const Complex>(in).subspan(at, n), std::span<const std::uint64_t>(delays).subspan(at, n), std::span<Complex>(got).subspan(at, n));
                }
                expect(that % std::ranges::equal(reference, got)) << std::format("order {}, chunk {} changed the stream", order, chunk);
            }
        }
    };

    "the accessors state the lag, and the history follows the delay"_test = [] {
        const gr::filter::ResamplerDesign design = designFractionalDelay(32UZ, kRolloff);
        FractionalDelayLine<Complex>      line(32UZ, 1, design.taps, 400ULL);

        expect(line.bankSize() == 32UZ);
        expect(line.order() == 1);
        expect(line.prototypeLength() == design.taps.size());
        expect(line.tapsPerArm() == (design.taps.size() + 31UZ) / 32UZ);
        expect(approx(line.groupDelaySamples(), 0.5 * static_cast<double>(design.taps.size() - 1UZ) / 32., 1e-12));
        expect(approx(line.latencySamples(), 1. + line.groupDelaySamples(), 1e-12)) << "the extra sample is the wrap arm's reach, and it is stated";
        expect(line.historySamples() == 400UZ + line.tapsPerArm() + 1UZ);
        expect(line.maxDelaySamples() == 400ULL);

        std::println("L = 32, q = 1 at a 60 dB design: {} taps, {} per arm, group delay {:.4f} samples, latency {:.4f}, history {} samples", design.taps.size(), line.tapsPerArm(), line.groupDelaySamples(), line.latencySamples(), line.historySamples());
    };

    "what the line refuses"_test = [] {
        const gr::filter::ResamplerDesign design = designFractionalDelay(8UZ, kRolloff);

        expect(throws<std::invalid_argument>([&] { (void)FractionalDelayLine<Complex>(0UZ, 1, design.taps); })) << "a bank with no arms";
        expect(throws<std::invalid_argument>([&] { (void)FractionalDelayLine<Complex>(8UZ, 2, design.taps); })) << "an order that is not 0, 1 or 3";
        expect(throws<std::invalid_argument>([&] { (void)FractionalDelayLine<Complex>(8UZ, 1, std::span<const float>{}); })) << "no taps";

        FractionalDelayLine<Complex> line(8UZ, 1, design.taps, 4ULL);
        const std::vector<Complex>   in(64UZ);
        std::vector<Complex>         out(64UZ);

        expect(throws<std::invalid_argument>([&] { line.process(in, std::vector<std::uint64_t>(63UZ), out); })) << "one delay short of the samples";
        expect(throws<std::invalid_argument>([&] { line.process(in, std::vector<std::uint64_t>(64UZ), std::span<Complex>(out).first(63UZ)); })) << "one output short";
        expect(throws<std::invalid_argument>([&] { line.process(in, std::vector<std::uint64_t>(64UZ, fractionalDelayQ32(5.)), out); })) << "a delay past the history the line was sized for";
        expect(nothrow([&] { line.process(in, std::vector<std::uint64_t>(64UZ, fractionalDelayQ32(4.)), out); })) << "and the bound itself is admitted";

        expect(throws<std::invalid_argument>([] { (void)fractionalDelayQ32(-1.); })) << "a negative delay is a read from the future";
        expect(throws<std::invalid_argument>([] { (void)fractionalDelayQ32(std::numeric_limits<double>::quiet_NaN()); }));
        expect(throws<std::invalid_argument>([] { (void)fractionalDelayQ32(static_cast<double>(kMaxFractionalDelaySamples)); })) << "2^31 samples is where the fixed point runs out";
        expect(nothrow([] { (void)fractionalDelayQ32(static_cast<double>(kMaxFractionalDelaySamples) - 1.); }));

        std::vector<std::uint64_t> two(2UZ);
        expect(throws<std::invalid_argument>([&] { fractionalDelayQ32(std::vector<double>{1e-3, 2e-3, 3e-3}, 48'000., two); })) << "the span conversion is paired too";
        expect(throws<std::invalid_argument>([&] { fractionalDelayQ32(std::vector<double>{1e-3, 2e-3}, -1., two); })) << "a rate that is not positive";
    };

    "the fixed point is exact where it says it is"_test = [] {
        expect(fractionalDelayQ32(0.) == 0ULL);
        expect(fractionalDelayQ32(1.) == kArbitraryOne);
        expect(fractionalDelayQ32(0.5) == kArbitraryOne / 2ULL);
        expect(fractionalDelayQ32(7.125) == 7ULL * kArbitraryOne + kArbitraryOne / 8ULL) << "an eighth is a whole number of Q32 steps";

        // The claim in the header: a `double`'s spacing is seven orders below what the representation keeps, so
        // two delays a `2^-32` sample apart land on adjacent fixed-point values rather than the same one.
        constexpr double kStep = 1. / 4'294'967'296.;
        expect(fractionalDelayQ32(3. + kStep) - fractionalDelayQ32(3.) == 1ULL);
    };
};

int main() { /* not needed for UT */ }
