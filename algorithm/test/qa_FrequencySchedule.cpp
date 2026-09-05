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

#include <gnuradio-4.0/algorithm/signal/Phasor.hpp>
#include <gnuradio-4.0/algorithm/timing/FrequencySchedule.hpp>
#include <gnuradio-4.0/algorithm/timing/SampleClock.hpp>

using namespace boost::ut;
using gr::signal::Phasor;
using gr::timing::FrequencySchedule;
using gr::timing::offsetFor;
using gr::timing::SampleClock;

namespace {

constexpr double kTwoPi = 2. * std::numbers::pi_v<double>;

/// The schedule's contract restated independently: clamped outside the table, and between two knots the
/// straight line written the other way round, so a shared slope cannot make both this and the kernel wrong
/// together.
[[nodiscard]] double oracleOffset(std::span<const std::int64_t> times, std::span<const double> offsets, std::int64_t t_ns) {
    if (t_ns <= times.front()) {
        return offsets.front();
    }
    if (t_ns >= times.back()) {
        return offsets.back();
    }
    const std::size_t index = static_cast<std::size_t>(std::upper_bound(times.begin(), times.end(), t_ns) - times.begin()) - 1UZ;
    const double      w     = static_cast<double>(t_ns - times[index]) / static_cast<double>(times[index + 1UZ] - times[index]);
    return offsets[index] * (1. - w) + offsets[index + 1UZ] * w;
}

/// The closed-form integral of that offset from `a` to `b`, in cycles: a trapezoid between consecutive
/// breakpoints, located by a search per breakpoint rather than by a cursor.
[[nodiscard]] double oracleCycles(std::span<const std::int64_t> times, std::span<const double> offsets, std::int64_t a, std::int64_t b) {
    double       cycles = 0.;
    std::int64_t x      = a;
    while (x < b) {
        const auto         next = std::upper_bound(times.begin(), times.end(), x);
        const std::int64_t end  = (next == times.end()) ? b : std::min(*next, b);
        cycles += 0.5 * (oracleOffset(times, offsets, x) + oracleOffset(times, offsets, end)) * static_cast<double>(end - x) * 1e-9;
        x = end;
    }
    return cycles;
}

/// Kahan summation, because the question is what the increments are, not how well a naive loop adds 48000 of
/// them: a plain sum of a 3e4 rad total loses more than the 1e-9 the increments themselves are held to.
[[nodiscard]] double compensatedSum(std::span<const double> values) {
    double sum          = 0.;
    double compensation = 0.;
    for (const double value : values) {
        const double adjusted = value - compensation;
        const double next     = sum + adjusted;
        compensation          = (next - sum) - adjusted;
        sum                   = next;
    }
    return sum;
}

/// A pass-shaped table: `nKnots` knots evenly spread over `spanNs`, following the S-curve a LEO Doppler profile
/// has — high on approach, through zero at closest approach, low going away.
struct Profile {
    std::vector<std::int64_t> times;
    std::vector<double>       offsets;
};

[[nodiscard]] Profile leoProfile(std::size_t nKnots, std::int64_t startNs, std::int64_t spanNs) {
    Profile profile;
    profile.times.resize(nKnots);
    profile.offsets.resize(nKnots);
    for (std::size_t i = 0UZ; i < nKnots; ++i) {
        const double u     = static_cast<double>(i) / static_cast<double>(nKnots - 1UZ);
        profile.times[i]   = startNs + static_cast<std::int64_t>(static_cast<double>(spanNs) * u);
        profile.offsets[i] = -10'000. * std::tanh(6. * (u - 0.5));
    }
    return profile;
}

} // namespace

const boost::ut::suite<"FrequencySchedule"> frequencyScheduleTests = [] {
    // Criterion 1. A ramp from -5 kHz to +5 kHz over ten seconds: the knots read their own values, the midpoint
    // and the quarter points read the interpolant exactly — the numbers are chosen so "exactly" is a fair word —
    // and outside the table the ends hold rather than continuing the line.
    "a ramp reads its knots, its interpolant and its held ends"_test = [] {
        const std::vector<std::int64_t> times{0LL, 10'000'000'000LL};
        const std::vector<double>       offsets{-5'000., 5'000.};
        const FrequencySchedule         schedule(times, offsets);

        expect(eq(schedule.size(), 2UZ));
        expect(eq(schedule.offsetAt(0LL), -5'000.));
        expect(eq(schedule.offsetAt(10'000'000'000LL), 5'000.));
        expect(eq(schedule.offsetAt(5'000'000'000LL), 0.)) << "the midpoint of a symmetric ramp";
        expect(eq(schedule.offsetAt(2'500'000'000LL), -2'500.));
        expect(eq(schedule.offsetAt(7'500'000'000LL), 2'500.));

        expect(eq(schedule.offsetAt(-1LL), -5'000.)) << "one nanosecond before the table";
        expect(eq(schedule.offsetAt(-1'000'000'000'000LL), -5'000.));
        expect(eq(schedule.offsetAt(std::numeric_limits<std::int64_t>::min()), -5'000.));
        expect(eq(schedule.offsetAt(10'000'000'001LL), 5'000.)) << "one nanosecond after it";
        expect(eq(schedule.offsetAt(std::numeric_limits<std::int64_t>::max()), 5'000.)) << "held, never extrapolated";

        for (std::int64_t t = -2'000'000'000LL; t <= 12'000'000'000LL; t += 37'000'011LL) {
            expect(approx(schedule.offsetAt(t), oracleOffset(times, offsets, t), 1e-9));
        }
    };

    // Criterion 2. The span deliberately starts before the table and ends after it, so it covers the held value,
    // both segments and the held value again, and every knot falls inside a sample interval rather than on its
    // boundary — which is the case the trapezoid split exists for.
    "the accumulated phase is the schedule's integral, however it is chunked"_test = [] {
        constexpr std::size_t kSamples = 48'000UZ;

        const std::vector<std::int64_t> times{200'000'003LL, 500'000'007LL, 800'000'011LL};
        const std::vector<double>       offsets{-5'000., 2'000., 4'000.};
        const FrequencySchedule         schedule(times, offsets);
        const SampleClock               clock(0ULL, 0LL, 48'000ULL, 1ULL);

        std::vector<double> whole(kSamples);
        schedule.phaseIncrementsFor(clock, 0ULL, whole);

        const double reference = kTwoPi * oracleCycles(times, offsets, clock.timeOf(0ULL), clock.timeOf(kSamples));
        const double achieved  = std::abs(compensatedSum(whole) - reference);
        std::println("qa_FrequencySchedule: accumulated phase {:.6f} rad over {} samples, {:.3e} rad from the closed-form integral", compensatedSum(whole), kSamples, achieved);
        expect(lt(achieved, 1e-9)) << "the accumulated phase is not the schedule's integral";

        // Same schedule, same clock, different call boundaries: the increments must be the same numbers, not
        // merely close ones, because a graph that changes its buffer size must not change its output.
        for (const std::size_t chunk : {1UZ, 7UZ, 1'000UZ, 12'345UZ}) {
            std::vector<double> pieced(kSamples);
            for (std::size_t produced = 0UZ; produced < kSamples; produced += chunk) {
                const std::size_t n = std::min(chunk, kSamples - produced);
                schedule.phaseIncrementsFor(clock, produced, std::span<double>(pieced).subspan(produced, n));
            }
            expect(that % std::ranges::equal(pieced, whole)) << std::format("chunk size {} changed the increments", chunk);
            expect(eq(compensatedSum(pieced), compensatedSum(whole)));
        }
    };

    // The cursor is an optimization, and an optimization has to give the same answer as the thing it replaces.
    // A thousand knots 20820 ns apart against a sample period of 20833 ns puts a knot inside very nearly every
    // sample interval, which is the case a cursor that moved only once per call would get wrong; the span runs
    // past the last knot as well, so the held tail is walked too.
    "the cursor walk agrees with a search per sample"_test = [] {
        constexpr std::size_t   kSamples = 1'200UZ;
        constexpr std::uint64_t kFirst   = 0ULL;

        const Profile           profile = leoProfile(1'000UZ, 0LL, 20'800'000LL);
        const FrequencySchedule schedule(profile.times, profile.offsets);
        const SampleClock       clock(0ULL, 0LL, 48'000ULL, 1ULL);

        std::vector<double> increments(kSamples);
        schedule.phaseIncrementsFor(clock, kFirst, increments);

        double worst = 0.;
        for (std::size_t k = 0UZ; k < kSamples; ++k) {
            const double want = kTwoPi * oracleCycles(profile.times, profile.offsets, clock.timeOf(kFirst + k), clock.timeOf(kFirst + k + 1UZ));
            worst             = std::max(worst, std::abs(increments[k] - want));
        }
        std::println("qa_FrequencySchedule: worst per-sample increment difference against a per-sample search {:.3e} rad", worst);
        expect(lt(worst, 1e-12));

        // The cursor is what a chunk boundary resets, so a dense table is where chunk independence is worth
        // asking about a second time.
        std::vector<double> oneAtATime(kSamples);
        for (std::size_t k = 0UZ; k < kSamples; ++k) {
            schedule.phaseIncrementsFor(clock, kFirst + k, std::span<double>(oneAtATime).subspan(k, 1UZ));
        }
        expect(that % std::ranges::equal(oneAtATime, increments)) << "a dense table read one sample at a time";
    };

    // Criterion 3's sign anchor. The spec quotes +10.207 kHz for this case; worked out it is
    // 437e6 * 7000 / 299792458 = 10203.7257 Hz, so the spec's figure is 3.3 Hz high and the correct one is
    // pinned here. What matters beyond the digits is the sign: closing reads high.
    "a closing pass reads high, to the hertz"_test = [] {
        const double closing = offsetFor(-7'000., 437e6);

        std::println("qa_FrequencySchedule: offsetFor(-7000 m/s, 437 MHz) = {:.4f} Hz", closing);
        expect(gt(closing, 0.)) << "a closing pass must read high";
        expect(eq(std::llround(closing), 10'204LL)) << "437e6 * 7000 / 299792458 = 10203.7257 Hz";
        expect(approx(closing, 10'203.7257, 1e-3));
        expect(eq(closing, 437e6 * 7'000. / 299'792'458.));

        expect(eq(offsetFor(7'000., 437e6), -closing)) << "a receding pass reads low by the same amount";
        expect(eq(offsetFor(0., 437e6), 0.));
        expect(approx(offsetFor(-1., 299'792'458.), 1., 1e-12)) << "one meter per second on a carrier of c hertz is one hertz";
    };

    "a table that cannot be interpolated is refused"_test = [] {
        const std::vector<std::int64_t> two{0LL, 1'000LL};
        const std::vector<double>       twoOffsets{0., 1.};

        expect(nothrow([&] { (void)FrequencySchedule(two, twoOffsets); })) << "two knots make a line";

        expect(throws<std::invalid_argument>([] {
            const std::vector<std::int64_t> times{0LL};
            const std::vector<double>       offsets{0.};
            (void)FrequencySchedule(times, offsets);
        })) << "one knot is not a schedule";
        expect(throws<std::invalid_argument>([] { (void)FrequencySchedule(std::span<const std::int64_t>{}, std::span<const double>{}); })) << "no knots at all";
        expect(throws<std::invalid_argument>([&] { (void)FrequencySchedule(two, std::span<const double>(twoOffsets).first(1UZ)); })) << "the two vectors must be paired";

        expect(throws<std::invalid_argument>([] {
            const std::vector<std::int64_t> times{0LL, 1'000LL, 1'000LL};
            const std::vector<double>       offsets{0., 1., 2.};
            (void)FrequencySchedule(times, offsets);
        })) << "two knots at the same time";
        expect(throws<std::invalid_argument>([] {
            const std::vector<std::int64_t> times{0LL, 2'000LL, 1'000LL};
            const std::vector<double>       offsets{0., 1., 2.};
            (void)FrequencySchedule(times, offsets);
        })) << "time running backwards";

        for (const double bad : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()}) {
            expect(throws<std::invalid_argument>([bad] {
                const std::vector<std::int64_t> times{0LL, 1'000LL};
                const std::vector<double>       offsets{0., bad};
                (void)FrequencySchedule(times, offsets);
            })) << "an offset that is not a frequency";
        }
    };

    "the table's ceiling is the spec's"_test = [] {
        expect(eq(FrequencySchedule::kMinKnots, 2UZ));
        expect(eq(FrequencySchedule::kMaxKnots, 1UZ << 20));

        const Profile full = leoProfile(FrequencySchedule::kMaxKnots, 0LL, 600'000'000'000LL);
        expect(nothrow([&] { (void)FrequencySchedule(full.times, full.offsets); }));

        std::vector<std::int64_t> tooMany(FrequencySchedule::kMaxKnots + 1UZ);
        std::vector<double>       tooManyOffsets(FrequencySchedule::kMaxKnots + 1UZ, 0.);
        for (std::size_t i = 0UZ; i < tooMany.size(); ++i) {
            tooMany[i] = static_cast<std::int64_t>(i);
        }
        expect(throws<std::invalid_argument>([&] { (void)FrequencySchedule(tooMany, tooManyOffsets); }));
    };

    // A constant schedule is the degenerate case the arithmetic must still get right: every increment is the
    // plain 2*pi*f/fs, and the trapezoid must not introduce a difference where the integral has none.
    "a flat schedule gives the plain increment"_test = [] {
        const std::vector<std::int64_t> times{0LL, 10'000'000'000LL};
        const std::vector<double>       offsets{1'000., 1'000.};
        const FrequencySchedule         schedule(times, offsets);
        const SampleClock               clock(0ULL, 0LL, 1'000'000ULL, 1ULL); // 1 MS/s: a sample is exactly 1000 ns

        std::vector<double> increments(64UZ);
        schedule.phaseIncrementsFor(clock, 5'000ULL, increments);

        for (const double increment : increments) {
            expect(approx(increment, kTwoPi * 1'000. / 1'000'000., 1e-15));
        }
    };

    // The composition a Doppler-correcting block is: a schedule's increments straight into the phasor that
    // rides them. Neither kernel is interesting alone — the schedule produces numbers nobody reads and the
    // phasor takes increments from anywhere — so the pair is what has to be shown working, at kernel level,
    // before a block puts a name on it.
    //
    // A two-knot ramp from -5 kHz to +15 kHz across two seconds at 48 kS/s. The mixed tone's angle is unwrapped
    // against the schedule's own closed-form integral and the residual is what is asserted, at every sample
    // rather than at the end: an error that grew with the run would show as a residual that grew, and an error
    // that did not grow is the property a coherent demodulator downstream depends on.
    //
    // The tolerance is float32's, and it is the output's, not the phasor's. The phase is accumulated in double
    // and reduced every step, so its own error over 96000 samples is a few times 1e-11 rad. What the test reads
    // is `atan2` of a `std::complex<float>` that came out of a complex multiply: four products and two sums,
    // each rounded at 2^-24 = 5.96e-8 relative, which bounds the angle of the result at roughly 8 * 2^-24 =
    // 4.8e-7 rad. The criterion is 1e-6 rad, twice that, and the achieved figure is printed.
    "a schedule's increments drive the phasor to the schedule's own integral"_test = [] {
        constexpr std::size_t kSamples   = 96'000UZ;
        constexpr double      kTolerance = 1e-6;

        const std::vector<std::int64_t> times{0LL, 2'000'000'000LL};
        const std::vector<double>       offsets{-5'000., 15'000.};
        const FrequencySchedule         schedule(times, offsets);
        const SampleClock               clock(0ULL, 0LL, 48'000ULL, 1ULL);

        std::vector<double> increments(kSamples);
        schedule.phaseIncrementsFor(clock, 0ULL, increments);

        // A constant unit-magnitude input, so that what comes out is the input's own angle plus the phasor's and
        // the multiply is exercised rather than stepped around. 0.6 and 0.8 are a unit vector to within their
        // own float rounding, and the reference takes the angle from the stored floats rather than from the
        // decimal that produced them.
        const std::complex<float>              carrier{0.6f, 0.8f};
        const std::vector<std::complex<float>> in(kSamples, carrier);
        std::vector<std::complex<float>>       out(kSamples);

        Phasor<float> phasor;
        phasor.configure(0., 0.);
        phasor.mixModulated(std::span<const double>(increments), in, out);

        const double       carrierAngle = std::atan2(static_cast<double>(carrier.imag()), static_cast<double>(carrier.real()));
        const std::int64_t startTime    = clock.timeOf(0ULL);

        double worst = 0.;
        double last  = 0.;
        for (std::size_t k = 0UZ; k < kSamples; ++k) {
            const double reference = carrierAngle + kTwoPi * oracleCycles(times, offsets, startTime, clock.timeOf(k));
            const double measured  = std::atan2(static_cast<double>(out[k].imag()), static_cast<double>(out[k].real()));
            const double residual  = std::remainder(measured - reference, kTwoPi);
            worst                  = std::max(worst, std::abs(residual));
            last                   = reference - carrierAngle;
        }

        std::println("qa_FrequencySchedule: the mixed tone follows the schedule's integral to {:.3e} rad over {} samples of {:.1f} rad total", worst, kSamples, last);
        expect(lt(worst, kTolerance)) << "the composition's phase left the schedule's integral";
        expect(gt(std::abs(last), 60'000.)) << "a run short enough that drift could hide is not a test of drift";

        // The increments carry the schedule and nothing else: the same phasor fed a flat increment must not
        // reproduce them, or the assertion above would pass on a kernel that ignored its input.
        std::vector<double>              flat(kSamples, kTwoPi * 5'000. / 48'000.);
        std::vector<std::complex<float>> flatOut(kSamples);
        Phasor<float>                    other;
        other.configure(0., 0.);
        other.mixModulated(std::span<const double>(flat), in, flatOut);
        expect(that % !std::ranges::equal(flatOut, out)) << "a constant offset is not this schedule";
    };

    "an empty output span is a no-op"_test = [] {
        const std::vector<std::int64_t> times{0LL, 1'000LL};
        const std::vector<double>       offsets{0., 1.};
        const FrequencySchedule         schedule(times, offsets);
        const SampleClock               clock(0ULL, 0LL, 48'000ULL, 1ULL);

        expect(nothrow([&] { schedule.phaseIncrementsFor(clock, 0ULL, std::span<double>{}); }));
    };
};

int main() { /* not needed for UT */ }
